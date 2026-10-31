"""에이전트 가드레일·커밋된 비밀값 검사(#311) — 글로만 지키던 규칙을 기계로 막는 층과, 그 층이 조용히 꺼지지 않았는지의 검사.

1. `.claude/settings.json`(커밋됨)의 deny — 경로 지정 없는 전체 스테이징(`add`의 전체 옵션·`.`·추적 파일 일괄, `commit`의 전체
   옵션), 강제 push, main 직접 push·삭제를 Claude 명령 수준에서 막는다(남이 스테이징·수정한 파일이 딸려 main에 머지된 사고 #226).
   `git -C 경로`·`git -c 설정` 같은 전역 옵션이 앞에 와도 막는다(`git * push …` 꼴). 개인 allow는 미추적
   `.claude/settings.local.json` 몫이고, Claude Code는 deny → ask → allow 순으로 판정해 deny가 이긴다(공식 문서 Configure permissions).
   이 테스트는 두 방향을 본다 — 막아야 할 명령이 막히는가, 평소 쓰는 명령이 잘못 막히지 않는가(오탐이 나면 사람이 deny를 지운다).
   매칭은 문서대로 흉내 낸다: 하위 명령(`&&`·`||`·`;`·`|`·`&`·줄바꿈으로 나눈 것) 각각의 전체 문자열, `*`는 아무 문자열, 끝의 ` *`는
   인자 없는 형태도 맞음. 따옴표 안의 구분자·`timeout` 같은 래퍼 제거는 흉내 내지 않는 근사다 — 표본 명령에 그런 경우를 넣지 않는다.
   **보안 경계가 아니다**(문서가 밝힌 한계). 못 막는 것: 인자 없는 `git push`(현재 브랜치가 main일 때), 묶은 짧은 옵션(`-uf`),
   `sh -c`·절대 경로 git — 이것들은 지시 규칙(rag-flow·AGENTS.md)으로 지킨다. 머지(`gh pr merge`)는 ask에 넣지 않는다(#311 결정 1).
2. 커밋된 설정의 비밀값 — 공개 저장소다. `.env*`·`docker-compose*.yml`·`config.py`의 비밀 키와 URL 안 비밀번호는 빈 값·
   **값 전체가** 자리표시(`[…]`·`<…>`·changeme 등)·변수 참조(`${X}`·`$X`로 시작)만 허용한다. 사유가 있는 예외는 ALLOWED에 값의
   해시까지 고정한다 — 키만 허용하면 값이 바뀌어도 계속 통과한다. 실패 메시지에는 **키 이름만** 싣고 값은 출력하지 않는다.
"""

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / '.claude' / 'settings.json'

MUST_BLOCK = [
    'git add -A',
    'git add --all',
    'git add .',
    'git add -- .',
    'git add rag/ .',
    'git add ./',
    'git add -u',
    'git add --update',
    'git add . && git commit -m "정리"',
    'git -C ../consult-agent-issue1 add -A',
    'git commit -am "정리"',
    'git commit -q -a -m "정리"',
    'git push --force',
    'git push origin feat/x --force-with-lease',
    'git push -f origin feat/x',
    'git push origin +feat/x',
    'git push -u origin main',
    'git push origin HEAD:main',
    'git push origin refs/heads/main',
    'git push origin --delete main',
    'cd ../wt && git push -q origin main',
    'git -C ../wt push origin main',
    'git -C ../wt push -f origin feat/x',
    'git -C ../wt push origin +feat/x',
    'git -C ../wt push origin HEAD:main',
    'git -c core.x=y push --force origin feat/x',
]
MUST_PASS = [
    'git add -- rag/turn/cache.py tests/test_guardrails.py',
    'git add -- .claude/skills/rag-flow/SKILL.md',
    'git commit -q -m "fix(#1): 캐시 키 정정" -- rag/turn/cache.py',
    'git commit -q -m "docs(#2): 정리" -- AGENTS.md 2>&1 | tail -1',
    'git push -q -u origin chore/issue-311-guardrails',
    'git push -q origin --delete chore/issue-311-guardrails',
    'git push -q -u origin fix/issue-12-format',
    'git push -u origin feat/maintenance',
    'git push origin HEAD:maintenance',
    'git push origin x:main-fix',
    'git status --short',
    'git fetch --prune origin',
    'git pull -q --ff-only origin main',
    'git worktree add -q -b feat/issue-9-x ../consult-agent-issue9 origin/main',
    'git -C ../wt log --oneline -1',
    'gh pr merge 313 --merge',
]

SECRET_KEY = re.compile(r'PASSWORD|PASSWD|SECRET|API_?KEY|TOKEN|PRIVATE_KEY|ACCESS_KEY|CREDENTIAL|_KEY$', re.I)
NOT_SECRET = re.compile(r'MAX_TOKENS|_TOKENS$', re.I)  # 토큰 '개수' 설정
PLACEHOLDER = re.compile(r'\[[^\]]*\]|<[^>]*>|change[-_ ]?me|your[-_\w]*|placeholder|x{3,}|dummy|example|EMPTY', re.I)
VARIABLE = re.compile(r'\$\{?\w+')
ALLOWED = {  # (파일, 키): (값 sha256 앞 12자, 사유) — 값이 바뀌면 다시 걸린다. 등록은 리뷰 대상이다
    ('docker-compose.local.yml', 'POSTGRES_PASSWORD'): (
        'a942b37ccfaf',
        'FE 로컬 개발 컨테이너의 이미지 기본값(배포 비밀값 아님)',
    ),
}
KEY_VALUE = re.compile(r'^\s*-?\s*["\']?([A-Za-z_][\w.\-]*)\s*[:=]\s*(.*?)\s*$')
URL_PASSWORD = re.compile(r'://[^\s:@/]+:(\S+)@')


def _rule_matches(rule: str, command: str) -> bool:
    if not (rule.startswith('Bash(') and rule.endswith(')')):
        return False
    pattern = rule[5:-1]
    regex = '.*'.join(re.escape(part) for part in pattern.split('*'))
    if re.fullmatch(regex, command):
        return True
    return pattern.endswith(' *') and pattern.count('*') == 1 and command == pattern[:-2]


def _blocked(rules: list[str], command: str) -> bool:
    separators = r'&&|\|\||;|\|&|\||(?<![>&])&(?!>)|\n'  # 리다이렉션 `2>&1`의 &는 구분자가 아니다
    parts = [p.strip() for p in re.split(separators, command) if p.strip()]
    return any(_rule_matches(rule, part) for rule in rules for part in parts)


def _deny() -> list[str]:
    return json.loads(SETTINGS.read_text(encoding='utf-8'))['permissions']['deny']


def test_deny가_있고_필수_명령을_막는다():
    leaked = [c for c in MUST_BLOCK if not _blocked(_deny(), c)]
    assert not leaked, f'.claude/settings.json deny가 막지 못한다: {leaked}'


def test_deny는_평소_명령을_막지_않는다():
    false_hits = [c for c in MUST_PASS if _blocked(_deny(), c)]
    assert not false_hits, f'deny 오탐 — 평소 쓰는 명령이 막힌다: {false_hits}'


def test_공유_설정에는_deny만_둔다():
    permissions = json.loads(SETTINGS.read_text(encoding='utf-8'))['permissions']
    assert set(permissions) == {'deny'}, '개인 allow·ask는 .claude/settings.local.json(미추적)에 둔다'


def test_매칭_흉내는_문서의_규칙을_따른다():
    assert _rule_matches('Bash(git log *)', 'git log')  # 끝의 ' *'는 인자 없는 형태도 맞는다
    assert not _rule_matches('Bash(git add .)', 'git add .claude/x')  # * 없으면 정확히 일치
    assert _blocked(['Bash(git add -A*)'], 'cd x && git add -A')  # 하위 명령마다 본다
    assert not _blocked(['Bash(x)'], 'git status 2>&1')  # 리다이렉션을 쪼개지 않는다


def _committed(pattern: str) -> list[Path]:
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / f for f in out.split('\0') if f and re.search(pattern, f) and (ROOT / f).is_file()]


def _safe(value: str) -> bool:
    value = value.strip().strip('"\'')
    return not value or bool(VARIABLE.match(value)) or bool(PLACEHOLDER.fullmatch(value))


def _allowed(rel: str, key: str, value: str) -> bool:
    entry = ALLOWED.get((rel, key))
    return bool(entry) and hashlib.sha256(value.strip().strip('"\'').encode()).hexdigest()[:12] == entry[0]


def _config_violations(rel: str, text: str) -> list[str]:
    bad = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith('#') or not (m := KEY_VALUE.match(line)):
            continue
        key, value = m.group(1), re.sub(r'\s+#.*$', '', m.group(2)).strip().strip('"\'')
        if SECRET_KEY.search(key) and not NOT_SECRET.search(key) and not _safe(value) and not _allowed(rel, key, value):
            bad.append(f'{rel}:{lineno} {key}')
        bad += [f'{rel}:{lineno} {key} URL 안 비밀번호' for p in URL_PASSWORD.findall(value) if not _safe(p)]
    return bad


def test_커밋된_설정에_평문_비밀값이_없다():
    bad = []
    for path in _committed(r'(^|/)\.env[^/]*$|(^|/)docker-compose[^/]*\.ya?ml$'):
        bad += _config_violations(str(path.relative_to(ROOT)), path.read_text(encoding='utf-8'))
    assert not bad, f'평문 비밀값(값은 생략) — 빈 값·자리표시·${{변수}}로 바꾸거나 사유·값 해시와 함께 ALLOWED에: {bad}'


def test_config의_비밀_필드_기본값은_비어_있다():
    tree = ast.parse((ROOT / 'config.py').read_text(encoding='utf-8'))
    bad = [
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and SECRET_KEY.search(node.target.id)
        and not NOT_SECRET.search(node.target.id)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and not _safe(node.value.value)
    ]
    assert not bad, f'config.py 비밀 필드에 기본값이 있다(값은 생략): {bad}'


def test_비밀값_판정_자체():
    ok = 'API_KEY=\nTOKEN=${T}\nDB=postgresql://u:[PASSWORD]@h/d\nKEY2=<채울 것>  # 주석\nMAX_TOKENS=4000\n'
    assert _config_violations('x.env', ok) == []
    bad = 'API_KEY=abc123\nDB=postgresql://u:s3/cr3t@h/d\nPASSWORD=pa$$word\nTOKEN=exampleabc\n- "SECRET=x1"\n'
    assert _config_violations('x.env', bad) == [
        'x.env:1 API_KEY',
        'x.env:2 DB URL 안 비밀번호',
        'x.env:3 PASSWORD',
        'x.env:4 TOKEN',
        'x.env:5 SECRET',
    ]
