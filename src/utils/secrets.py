import os
import json


def _read_dotenv_file(path):
    """Read a simple .env file and return dict of key->value.
    Simple parser: KEY=VALUE, ignores lines starting with # and empty lines.
    Values may be quoted with single or double quotes.
    """
    env = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"\'')
                env[k] = v
    except Exception:
        pass
    return env


def load_api_key(var_name='GEMINI_API_KEY'):
    """Load API key from environment, .env file, or secrets.json.

    Order:
    1. Environment variable (os.environ)
    2. .env file (try python-dotenv then fallback to simple parser) in a set of likely locations
    3. secrets.json in likely locations

    When a .env is found it will be loaded into os.environ (so later code sees it during startup).
    Returns the key string or None if not found.
    """
    # 1) environment
    key = os.environ.get(var_name)
    if key:
        return key

    # Determine useful paths to search for .env and secrets.json
    this_dir = os.path.dirname(__file__)
    repo_root = os.path.normpath(os.path.join(this_dir, '..'))
    possible_dotenv_paths = [
        os.path.join(os.getcwd(), '.env'),
        os.path.join(repo_root, '.env'),
        os.path.join(this_dir, '.env'),
    ]

    # 2) try python-dotenv if available (and load env into os.environ)
    try:
        from dotenv import load_dotenv
        for p in possible_dotenv_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                load_dotenv(p, override=False)
        # re-check environment after loading
        key = os.environ.get(var_name)
        if key:
            return key
    except Exception:
        # python-dotenv not available, fall back to simple parser
        for p in possible_dotenv_paths:
            p = os.path.normpath(p)
            if os.path.exists(p):
                envmap = _read_dotenv_file(p)
                if var_name in envmap:
                    # set in os.environ for subsequent imports
                    os.environ.setdefault(var_name, envmap[var_name])
                    return envmap[var_name]

    # 3) secrets.json fallback
    possible_json = [
        os.path.join(os.getcwd(), 'secrets.json'),
        os.path.join(repo_root, 'secrets.json'),
        os.path.join(this_dir, '..', 'secrets.json')
    ]
    for p in possible_json:
        p = os.path.normpath(p)
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if var_name in data and data[var_name]:
                        os.environ.setdefault(var_name, data[var_name])
                        return data[var_name]
            except Exception:
                pass

    return None
