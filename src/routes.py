from typing import Dict, List
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.param_functions import Form, Query
from src.utils import render
from src.model.main import get_images, get_results, plot_graphisc
router: APIRouter = APIRouter()
 


@router.get('/lab2')
async def lab2() -> Dict[str, str]:
    return render('templates/lab2.html')





@router.post('/api/lab2')
async def api_lab2(poly_type:str=Form(...),degrees:str = Form(...), file:UploadFile= File(...),dimensions:str = Form(...)):
    result = get_results(file,degrees,poly_type,dimensions)
    return {'result':result}