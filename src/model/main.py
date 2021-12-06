import numpy as np
import json
from scipy.special.orthogonal import hermite
from src.model.model import Model, ResultPrinter
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
def get_results(file,degrees,poly_type,dimensions):
    data = pd.read_csv(BytesIO(file.file.read()),sep='\t')
    #data = data.drop('q',axis=1)
    degrees_list = [int(degree) for degree in degrees.split(",")]
    dimensions_list = [int(dim) for dim in dimensions.split(",")]
    model = Model(data=data,degrees = degrees_list,polynom_type=poly_type,dimensions=dimensions_list,split_lambda=True) 
    model.build()
    results_printer = ResultPrinter(model)
    graphics = get_images(data,model,dimensions_list)
    file = write_results_to_file(model.F,
    model.get_metric(),
    dict(results_printer.form_Psi()),
    dict(results_printer.form_Fi()),
    dict(results_printer.form_Y()),
    dict(results_printer.form_standart_Y()))
    result ={"Y_pred":model.F.tolist(),
    "norm_errors":model.get_metric().tolist(),
    "Psi":dict(results_printer.form_Psi()),
    "Fi":dict(results_printer.form_Fi()),
    "Y":dict(results_printer.form_Y()),
    "Y_standart":dict(results_printer.form_standart_Y()),
    "graphics":graphics,
    "file_results":file}
    return result

def get_images(data:pd.DataFrame,model:Model,dimensions:list):
    shift =dimensions[2] + dimensions[1] + dimensions[0]
    Y_denormed =  data.iloc[:, shift:].to_numpy()
    print(Y_denormed)
    Y_normed = model.Y
    Y_pred_normed = model.F
    Y_pred_denormed = model.F *(Y_denormed.max(axis=0) -Y_denormed.min(axis=0)) + Y_denormed.min(axis=0)
    graphics = plot_graphisc(Y_denormed,Y_normed,Y_pred_normed,Y_pred_denormed)
    return graphics

def plot_graphisc(Y_denormed,Y_normed,Y_pred_normed,Y_pred_denormed):
    subplots = []
    for i in range(Y_denormed.shape[1]):
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        ax[0][0].plot(Y_denormed[:,i],label = "Y{i} denormed".format(i=i+1))
        ax[0][0].plot(Y_pred_denormed[:,i],label = "Y{i} pred denormed".format(i=i+1))
        ax[0][0].set(title="Denormed graphics")
        ax[0][0].legend()
        ax[0][1].plot(np.abs(Y_normed[:,i]-Y_pred_normed[:,i]),label ="Y{i} norm error".format(i=i+1))
        ax[0][1].set(title="Normed errors")
        ax[0][1].legend()
        ax[1][0].plot(Y_normed[:,i],label ="Y{i} normed".format(i=i+1))
        ax[1][0].plot(Y_pred_normed[:,i],label ="Y{i} pred normed".format(i=i+1))
        ax[1][0].set(title="Normed graphics")
        ax[1][0].legend()
        ax[1][1].plot(np.abs(Y_denormed[:,i]-Y_pred_denormed[:,i]),label ="Y{i} denormed error".format(i=i+1))
        ax[1][1].set(title="Denormed errors")
        ax[1][1].legend()
        print("hermite")
        buf = BytesIO()
        fig.savefig(buf,format='png')
       
        buf.seek(0)
        subplots.append(base64.b64encode(buf.read()))
        plt.close()
        
    
    return subplots 


def write_results_to_file(Y_pred,metrics,Psi,Fi,Y,Y_standarts):
    buf = BytesIO()
    Y_pred_top = ""
    for i in range(Y_pred.shape[1]):
        Y_pred_top+="Y{i}\t".format(i=i+1)
    Y_pred_top+="\n"
    buf.write(Y_pred_top.encode())
    np.savetxt(buf,Y_pred,fmt='%1.3f')
    buf.write(b'Metrics:\n')
    np.savetxt(buf,metrics,fmt='%1.3f')
    buf.write(b'Psi:\n')
    buf.write(json.dumps(Psi,indent=4).encode())
    buf.write(b'Fi:\n')
    buf.write(json.dumps(Fi,indent=4).encode())
    buf.write(json.dumps(Y,indent=4).encode())
    buf.write(json.dumps(Y_standarts,indent=4).encode())
    buf.seek(0)
    encode_file = base64.b64encode(buf.read())
    return encode_file
