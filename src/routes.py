from typing import Dict
from fastapi import APIRouter
from src.utils import render

router: APIRouter = APIRouter()
state: int = 0


@router.get('/lab2')
async def lab2() -> Dict[str, str]:
    return render('templates/lab2.html')


@router.get('/api/lab2')
async def api_lab2() -> Dict[str, str]:
    global state
    state += 1
    return {'state': state}
