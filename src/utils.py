from fastapi.responses import HTMLResponse


def render(filename: str, status_code: int = 200) -> HTMLResponse:
    with open(filename, 'r') as f:
        return HTMLResponse(content=f.read(), status_code=status_code)
