"""Reference implementation of the two tasks, their state descriptions and the interface interventions.

Phase task. Agents hold phases theta_i on a circle. Each agent reads a description of the other agents'
positions relative to itself, answers "advance", "stay" or "retard", and all phases are updated at once:

    theta_i(t+1) = theta_i(t) + omega_i + K * a_i(t)

Reflection intervention. A mirrored description reverses the relative-phase histogram, which exchanges ahead
and behind. The returned action is then sign-reversed before the update. Mirror coins are fair. They are drawn
independently for each agent ("independent"), once for the whole population ("shared"), or never ("native").
In the shuffled control the mapped actions are permuted among agents before the update.

Resource task. Agents occupy one of three identical resources. Each agent reads how many other agents use each
resource and which one it uses, and names the resource it chooses. All agents decide from the same counts.
Label intervention. A permutation p of the three labels is applied before the description: displayed label j
stands for physical resource p[j]. The answer is mapped back through p. Permutations are drawn independently
for each agent ("independent") or once for all agents ("shared").

`verify_protocol.py` checks this module against every stored prompt, response and state transition.
The functions `reflection_flags` and `label_frames` state the sampling rule of the schedules. The recorded
experiments used the schedules stored in each design.json, which the verification reads directly.
"""
from pathlib import Path
import gzip, hashlib, json, re, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from circlemap.observation import RelativePhaseHistogram                                    # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram            # noqa: E402

BINS = 24
MOMENTS = 'moments_m1_m3'
PHASE_ACTIONS = {'retard': -1, 'stay': 0, 'advance': 1}
LABELS = 'ABC'


# ----------------------------------------------------------------------------- general helpers
def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_journal(path):
    """Yield the records of an append-only journal; a gzip-compressed copy (path + '.gz') is read transparently."""
    path = Path(path)
    opener = (lambda: open(path, encoding='utf-8')) if path.exists() else (lambda: gzip.open(str(path) + '.gz', 'rt', encoding='utf-8'))
    with opener() as handle:
        for line in handle:
            yield json.loads(line)


def terminal_results(path):
    """Final result of every task: {task_id: record}."""
    return {r['task_id']: r for r in read_journal(path) if r.get('event') == 'result' and r.get('terminal')}


# ----------------------------------------------------------------------------- phase task
def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def peer_histograms(theta, bins=BINS):
    """Row i is the normalized histogram of the other agents' phases relative to agent i (bins on [-pi, pi))."""
    p = np.asarray(theta, float)
    n = p.size
    delta = wrap(p[None, :] - p[:, None])
    ids = np.floor(np.round((delta + np.pi)*bins/(2*np.pi), 12)).astype(int) % bins
    mask = ~np.eye(n, dtype=bool)
    row = np.broadcast_to(np.arange(n)[:, None], (n, n))
    return np.bincount((row*bins + ids)[mask], minlength=n*bins).reshape(n, bins)/(n - 1)


def mirror_histogram(h):
    """Reflection of the relative-phase histogram: bin order reversed."""
    return np.asarray(h)[::-1]


def phase_prompt(h, representation=MOMENTS):
    """Text shown to one agent for the histogram h of its peers (default: first three circular moments)."""
    peers = int(round(1/np.min(np.asarray(h)[np.asarray(h) > 0]))) if np.any(np.asarray(h) > 0) else 16
    del peers  # the recorded populations have 16 peers; the count is fixed below as in the recorded prompts
    hist = RelativePhaseHistogram(edges=np.linspace(-np.pi, np.pi, BINS + 1), fractions=np.asarray(h), peer_count=16)
    return build_representation_prompt_from_histogram(representation, hist)


def parse_phase_action(raw):
    """Accept one JSON object with the single key "social_action", optionally inside one complete code fence."""
    text = raw.strip()
    match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'social_action'} or obj['social_action'] not in PHASE_ACTIONS:
        raise ValueError('invalid response')
    return PHASE_ACTIONS[obj['social_action']]


def reflection_flags(policy, n, rng):
    """Mirror coins for one step: none (native), one fair coin per agent (independent) or one fair coin for all (shared)."""
    if policy == 'native':
        return np.zeros(n, bool)
    if policy == 'independent':
        return rng.integers(0, 2, n).astype(bool)
    if policy == 'shared':
        return np.repeat(bool(rng.integers(0, 2)), n)
    raise ValueError(policy)


def physical_actions(raw_values, mirrored):
    """Map returned actions back: an action given to a mirrored description is sign-reversed."""
    return np.where(np.asarray(mirrored, bool), -np.asarray(raw_values), np.asarray(raw_values))


def phase_update(theta, omega, coupling, actions):
    return np.asarray(theta, float) + np.asarray(omega, float) + coupling*np.asarray(actions)


def order_parameter(theta, m=1):
    return np.abs(np.exp(1j*m*np.asarray(theta, float)).mean(axis=-1))


def phase_step(theta, omega, coupling, mirrored, respond, permutation=None):
    """One synchronous update. `respond(prompt_text)` returns the raw model output for one agent."""
    h = peer_histograms(theta)
    prompts = [phase_prompt(mirror_histogram(h[i]) if mirrored[i] else h[i]) for i in range(len(h))]
    raw = np.array([parse_phase_action(respond(text)) for text in prompts])
    actions = physical_actions(raw, mirrored)
    if permutation is not None:                       # shuffled control: agent i receives the action of agent permutation[i]
        actions = actions[np.asarray(permutation)]
    return phase_update(theta, omega, coupling, actions), dict(prompts=prompts, raw_values=raw, actions=actions)


# ----------------------------------------------------------------------------- resource task
RESOURCE_TEMPLATE = ('You choose one of three physically identical resources, labeled A, B, C. The labels carry no preferences. '
                     'The following counts include the other 16 agents, excluding you. If you choose a resource, its resulting load '
                     'is that count plus one. Your objective is to minimize the resulting load on the resource you choose. '
                     'You may stay on your current resource or switch; there is no switching cost.\n'
                     'Other-agent counts: A={a}, B={b}, C={c}.\nYour current resource: {own}.\n'
                     'Return only a JSON object with one key "resource" and value "A", "B", or "C".')


def resource_prompt(displayed_counts, displayed_own):
    a, b, c = (int(v) for v in displayed_counts)
    return RESOURCE_TEMPLATE.format(a=a, b=b, c=c, own=LABELS[int(displayed_own)])


def parse_resource_choice(raw):
    """Accept a complete initial JSON object {"resource": "A"|"B"|"C"}, optionally fenced; later text is ignored."""
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    match = re.match(r'\{.*?\}', text, flags=re.DOTALL)
    if not match:
        raise ValueError('invalid response')
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or set(obj) != {'resource'} or obj['resource'] not in list(LABELS):
        raise ValueError('invalid response')
    return LABELS.index(obj['resource'])


def label_frames(policy, n, rng):
    """Label permutations for one step: one random permutation per agent (independent) or one for all agents (shared)."""
    if policy == 'independent':
        return np.array([rng.permutation(3) for _ in range(n)])
    if policy == 'shared':
        return np.repeat(rng.permutation(3)[None, :], n, axis=0)
    raise ValueError(policy)


def displayed_state(state, agent, permutation):
    """Counts of the other agents and the agent's own resource as displayed under `permutation`."""
    state = np.asarray(state)
    p = np.asarray(permutation)
    counts = np.bincount(np.delete(state, agent), minlength=3)
    return counts[p], int(np.where(p == state[agent])[0][0])


def physical_choice(permutation, displayed_choice):
    return int(np.asarray(permutation)[displayed_choice])


def resource_step(state, frames, respond):
    """One synchronous update of all agents from the same counts. An invalid answer keeps the current resource."""
    state = np.asarray(state)
    new = state.copy()
    prompts = []
    for agent in range(len(state)):
        counts, own = displayed_state(state, agent, frames[agent])
        text = resource_prompt(counts, own)
        prompts.append(text)
        try:
            new[agent] = physical_choice(frames[agent], parse_resource_choice(respond(text)))
        except (ValueError, json.JSONDecodeError):
            pass
    return new, dict(prompts=prompts)


def normalized_congestion(state, resources=3):
    """Sum of squared loads divided by its minimum (97 for 17 agents and three resources)."""
    state = np.asarray(state)
    n = state.shape[-1]
    loads = np.stack([(state == j).sum(axis=-1) for j in range(resources)], axis=-1)
    q, r = divmod(n, resources)
    return (loads**2).sum(axis=-1)/(r*(q + 1)**2 + (resources - r)*q**2)
