"""Logged, cached, parallel chat calls. Every call records the model string the SERVER returns.
If you already have arc_logger.py from Part 1, keep its record fields; this module is a drop-in superset."""
import os, json, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor
import config

_lock = threading.Lock()
_sems = {}
_clients = {}
_cache = None

def _sem(host):
    k = "arc" if host == "arc" else host
    if k not in _sems:
        _sems[k] = threading.Semaphore(config.MAX_CONCURRENT_ARC if host == "arc" else config.MAX_CONCURRENT_OTHER)
    return _sems[k]

def _client(model_name):
    from openai import OpenAI
    if model_name not in _clients:
        c = config.MODELS[model_name]
        _clients[model_name] = OpenAI(base_url=os.environ[c["base_url_env"]], api_key=os.environ[c["key_env"]])
    return _clients[model_name]

def key_of(model_name, setting, prompt):
    return hashlib.sha1(f"{model_name}|{setting}|{config.TEMPERATURE}|{config.MAX_TOKENS}|{prompt}".encode()).hexdigest()

def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(config.LOG_PATH):
            for line in open(config.LOG_PATH):
                r = json.loads(line)
                if not r.get("error"):
                    _cache[r["key"]] = r
    return _cache

def call_model(model_name, setting, prompt, item_id="", experiment="", run=0):
    cache = _load_cache()
    k = key_of(model_name, setting, prompt)
    if k in cache:
        return cache[k]
    c = config.MODELS[model_name]
    model_id = c["variants"][setting]
    rec = dict(key=k, ts=time.time(), experiment=experiment, run=run, model_name=model_name, setting=setting,
               host=c["host"], requested_model=model_id, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS,
               item_id=item_id, prompt=prompt, response=None, returned_model=None, finish_reason=None,
               prompt_tokens=None, completion_tokens=None, reasoning_tokens_reported=None, seconds=None, error=None)
    t0 = time.time()
    with _sem(c["host"]):
        for attempt in range(4):
            try:
                r = _client(model_name).chat.completions.create(
                    model=model_id, messages=[{"role": "user", "content": prompt}],
                    temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
                ch = r.choices[0]
                rec.update(response=ch.message.content or "", finish_reason=ch.finish_reason, returned_model=r.model,
                           prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
                d = getattr(r.usage, "completion_tokens_details", None)
                rec["reasoning_tokens_reported"] = getattr(d, "reasoning_tokens", None) if d else None
                rec["error"] = None
                break
            except Exception as e:  # retry with backoff
                rec["error"] = repr(e)
                time.sleep(2 ** attempt)
    rec["seconds"] = time.time() - t0
    with _lock:
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
        with open(config.LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not rec["error"]:
            cache[k] = rec
    return rec

def run_batch(model_name, setting, jobs, experiment="", run=0, workers=10):
    """jobs: list of (item_id, prompt). Returns records in the same order."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(call_model, model_name, setting, p, i, experiment, run) for i, p in jobs]
        return [f.result() for f in futs]
