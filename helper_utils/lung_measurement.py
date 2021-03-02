import numpy as np
from scipy.signal import find_peaks
import joblib
import pandas as pd
  
def compute_keypoints(vals, display=False): 
    
    x = np.arange(len(vals))
    
    # find peaks
    peak_idxs = find_peaks(vals, distance=len(vals)//10)[0]
    peaks = np.array([vals[i] for i in peak_idxs])
    
    if len(peaks) < 3:
        print('not enough peaks computed, bad signal')
        return [-1] * 4

    # find minima
    minima_idxs = find_peaks(-vals, distance=len(vals)//10)[0]
    minimas = np.array([vals[i] for i in minima_idxs])
    
    # start of tidal 
    start = (0, vals[0])
    
    # end of exhale
    exhale_end_idx = peaks.argsort()[-1]
    exhale_end = (peak_idxs[exhale_end_idx], peaks[exhale_end_idx])
    
    # start of exhale 
    minima_coords = [(x, y) for x, y in zip(minima_idxs, minimas) if x < exhale_end[0]]
    minima_coords.sort(key=lambda x: x[1])
    exhale_start = minima_coords[0]
    
    # end of tidal
    peak_coords = [(x, y) for x, y in zip(peak_idxs, peaks) if x < exhale_end[0]]
    x_diffs = np.array([abs(x - exhale_end[0]) for x, _ in peak_coords])
    min_idx = x_diffs.argsort()[0]
    tidal_end = peak_coords[min_idx]
    
    if display: 
        
        plt.figure(1)
        plt.figure(figsize=(10,5))
        plt.plot(depth_chest_displacement, '-')
        plt.legend(['smoothed'], loc='best')

        # annotate effort points
        plt.annotate('start of tidal', start, 
                    xytext=(start[0] + 10, start[1] + .002), arrowprops=dict(arrowstyle="->"))
        plt.annotate('end of tidal', tidal_end,
                    xytext=(tidal_end[0] + 10, tidal_end[1] + .002), arrowprops=dict(arrowstyle="->"))
        plt.annotate('start of exhale', exhale_start,
                    xytext=(exhale_start[0] + 10, exhale_start[1] + .002), arrowprops=dict(arrowstyle="->"))
        plt.annotate('end of exhale', exhale_end,
                    xytext=(exhale_end[0] + 10, exhale_end[1] + .002), arrowprops=dict(arrowstyle="->"))

        plt.show()
    
    return start, tidal_end, exhale_start, exhale_end  

def compute_pft_measures(exhalation, flow_volume, flow_volume_exhalation):
    
    result = []
    
    # FVC = Volume After Blast - Volume After Inhalation
    FVC = exhalation.max() - 0
    print('FVC: ', round(FVC, 2)) 
    
    result.append(FVC)

    # FEV1 = Volume of air exhaled after one second (during exhalation blast)
    # signal is downsampled to 30 fps, so 30 frames = 1s 
    FEV1 = exhalation[6]
    print('FEV1: ', round(FEV1, 2))
    
    result.append(FEV1)

    # FEV1 / FVC 
    fev1_fvc_ratio = FEV1 / FVC
    print('FEV1 / FVC: ', round(fev1_fvc_ratio * 100, 2))
    
    result.append(fev1_fvc_ratio)

    # PEF = Maximum Flow 
    PEF = flow_volume.max()
    print('PEF: ', PEF)
    
    result.append(PEF)

    # FEF_25 = Flow of exhaled air at 25% FVC 
    fef_25_idxs = np.where(exhalation >= FVC * .25)[0]
    FEF_25 = flow_volume_exhalation[fef_25_idxs[1]]
    print('FEF_25: ', FEF_25)
    
    result.append(FEF_25)

    # FEF_50 = Flow of exhaled air at 50% FVC 
    fef_50_idxs = np.where(exhalation >= FVC * .50)[0]
    FEF_50 = flow_volume_exhalation[fef_50_idxs[1]]
    print('FEF_50: ', FEF_50)
    
    result.append(FEF_50)

    # FEF_75 = Flow of exhaled air at 75% FVC 
    fef_75_idxs = np.where(exhalation >= FVC * .75)[0]
    FEF_75 = flow_volume_exhalation[fef_75_idxs[1]]
    print('FEF_75: ', FEF_75)
    
    result.append(FEF_75)

    # FEF_25_75 = Mean forced exp flow between .25 and .75
    # (.75*FVC - .25*FVC) - (time(.25FVC) - time(.75FVC))
    FEF_25_75 = (.75*FVC - .25*FVC) / (abs(fef_25_idxs[1] - fef_75_idxs[1]))
    print('FEF_25_75: ', FEF_25_75)
    
    result.append(FEF_25_75)
    
    return result

def translate_chest_to_lung_params(chest_displacement, model_dir): 

    # load model
    lg = joblib.load(model_dir)

    # compute FVC of depth graph 
    depth_sensor_start, depth_sensor_tidal_end, depth_sensor_exhale_start, depth_sensor_exhale_end = compute_keypoints(np.array(chest_displacement))
    depthFVC = depth_sensor_exhale_end[1] - depth_sensor_exhale_start[1]

    # set info for model prediction 
    df_result = pd.DataFrame(columns=['DepthFVC', 'Height', 'Weight'])
    df_result.loc[len(df_result)] = [depthFVC, 70, 155]

    # predict 
    pred_results = lg.predict(df_result)
    predictedFVC = pred_results[0][0]

    # get depth graph from predicted FVC 
    final_test_volume = (chest_displacement - chest_displacement[depth_sensor_exhale_start[0]]) * (predictedFVC / depthFVC)

    # compute depth sensor lung params 
    test_exhalation = final_test_volume[depth_sensor_exhale_start[0]:depth_sensor_exhale_end[0]]
    test_flow = np.gradient(final_test_volume, .1)
    test_flow_exhale = test_flow[depth_sensor_exhale_start[0]:depth_sensor_exhale_end[0]]
    test_pft_results = compute_pft_measures(test_exhalation, test_flow, test_flow_exhale)
    
    return test_pft_results

