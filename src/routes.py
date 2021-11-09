from typing import Dict
from fastapi import APIRouter

router = APIRouter()


@router.get('/')
async def get() -> Dict[str, str]:
    return {'status': 'ok'}
