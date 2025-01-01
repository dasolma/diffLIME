import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from phm_framework.data import synthetic, meta
   
def plot_importances(signal, point_importances, prediction_score, local_pred, 
                     source_prob, centroids, title="", topi=10, base_frec=1 ):
    
    
    smeta = synthetic.generate_meta(signal, centroids)
    
    time = np.linspace(0, signal.shape[0], signal.shape[0])
    
    ini_importances = np.copy(point_importances)
    ini_importances  = ini_importances / np.abs(ini_importances).sum()
    
    point_importances = np.abs(point_importances)[:signal.shape[0]]
    
    # Función para suavizar las importancias usando una media móvil
    def smooth_importances(importancias, window_size=3):
        return np.convolve(importancias, np.ones(window_size) / window_size, mode='same')

    # Suavizar las importancias
    point_importances = smooth_importances(point_importances, window_size=10)

    # Normalizamos las importancias para que varíen entre 0 y 1
    norm_importancias = (point_importances - np.min(point_importances)) / (np.max(point_importances) - np.min(point_importances))
    
    top_arg = np.argsort(np.abs(ini_importances[:signal.shape[0]]))[-topi:]
    

    # Create the figure with subplots
    fig = plt.figure(figsize=(16, 6))  # Increase width for additional space
    ax_main = plt.subplot2grid((2, 5), (0, 2), colspan=3, rowspan=2)  # Main signal plot
    ax_freq = plt.subplot2grid((2, 5), (1, 1))             # Frequenies
    ax_env = plt.subplot2grid((2, 5), (0, 1))             # Envelope
    ax_freq_imp = plt.subplot2grid((2, 5), (0, 0), rowspan=2)             # Frequency importances plot

    
    # Usar un colormap para representar las importancias en el fondo
    cmap = matplotlib.colormaps['viridis']  # Colormap

    # Dibujar la señal en primer plano
    ax_main.plot(time, signal, label='Signal', color='black', linewidth=2)

    top_imp = ini_importances[top_arg]
    factor = 1000 if np.abs(ini_importances).max() >= 0.1 else 10000
    for x, y, s in zip(top_arg[np.where(top_imp >= 0)], 
                    signal[top_arg][np.where(top_imp >= 0)], 
                       top_imp[np.where(top_imp >= 0)]):
        ax_main.scatter(x, y, 
                    c="#5fad4eff", s=s*factor, marker="^", label="Positive Importance")
    
    
    for x, y, s in zip(top_arg[np.where(top_imp < 0)], 
                    signal[top_arg][np.where(top_imp < 0)], 
                       top_imp[np.where(top_imp < 0)]):
        ax_main.scatter(x, y, 
            c="#c73434ff", s=np.abs(s)*factor, marker="v", label="Negative Importance")
    
    # Crear un fondo de color según la importancia
    width = np.max(signal) - np.min(signal)
    im = ax_main.imshow(norm_importancias.reshape(1, -1), aspect='auto', cmap=cmap, 
               extent=[time[0], time[-1], np.min(signal) - width*0.1, np.max(signal) + width*0.1], alpha=0.3)  # Fondo con colores

    # Añadir la información al lateral izquierdo
    text = f"Prediction Score: {float(prediction_score):.2f}\n\nLocal Prediction: {float(local_pred):.2f}\n\nSource Probability: {float(source_prob):.2f}"
    ax_main.text(1, np.max(signal) + width*0.085, text, 
             fontsize=13, va='top', ha='left', 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Frecuencies plot and importances
    frec, magn = list(zip(*meta.extract_top_frequencies(signal, top_n=5)))
    frec = [f*base_frec for f in frec]
    for i, (freq, mag) in enumerate(zip(frec, magn)):
        ax_freq.vlines(x=freq, ymin=0, ymax=mag, colors='#96d9caff', linestyles='-', lw=2, label=f'Frequency: {freq} Hz' if mag == max(magn) else None)
        ax_freq.text(freq, mag + 0.2, f"F{i+1}", ha='center', fontsize=14, color='black')  # Add label above the line
    
    # Scatter points at the tip of the lines for clarity
    ax_freq.scatter(frec, magn, color='#399d87ff', zorder=5, label='Magnitudes')
    
    ax_freq.set_xlabel('Top 5 frequencies', fontsize=13)
    ax_freq.set_ylabel('Magnitude', fontsize=13)
    
    # Frequency importances (last 5 elements of the importance vector)
    frequency_importances = ini_importances[signal.shape[0]:signal.shape[0]+5]
    frequency_labels = [f"F{i+1}" for i in range(len(frequency_importances))]

    
    
    attributtes = (#[f"S{i}" for i in range(X.shape[1])] + 
               [f"F{i+1}" for i in range(5)] + 
               ['slope', 'noise', 'entropy', 'periodicity'] + 
               [f"E{i+1}" for i in range(len(centroids))])

    
    
    
    ini_importances = ini_importances[signal.shape[0]:]
    aux = list(zip(ini_importances, attributtes))
    frequency_importances, frequency_labels = list(zip(*sorted(aux, key=lambda x: -np.abs(x[0]))[:][::-1]))
        
    colors = ['#5fad4eff' if imp > 0 else '#c73434ff' for imp in frequency_importances]
    
    ax_freq_imp.barh(frequency_labels, frequency_importances,  color=colors, alpha=0.7)
    ax_freq_imp.set_title(f"Importances ({round(np.abs(ini_importances).sum() * 100)}%)", fontsize=16)
    ax_freq_imp.set_ylabel("Features", fontsize=13)
    #ax_freq_imp.set_xlim(-0.5, 0.5)
    
    # Etiquetas y título
    color_bar = fig.colorbar(im, ax=ax_main, label='Importance')  # Barra de color para la importancia
    ax_main.set_xlabel('Time', fontsize=13)
    ax_main.set_ylabel('Signal value', fontsize=13)
    ax_main.set_title('Time-Domain Signal Prediction Explanation', fontsize=16)
    
    # Plot envelope
    ax_env.plot(centroids[smeta[-1]][0], linewidth=2)
    ax_env.plot(centroids[smeta[-1]][1], linewidth=2)
    ax_env.fill_between(
        np.arange(len(centroids[smeta[-1]][0])),  # El eje X (puede ser el rango de los índices de los datos)
        centroids[smeta[-1]][0],  # Valores para la primera línea
        centroids[smeta[-1]][1],  # Valores para la segunda línea
        color='lightblue',        # Color del relleno
        alpha=0.5,                # Transparencia del relleno
        linewidth=0               # Opcional, para evitar líneas de borde
    )
    ax_env.set_title(f'Signal envelope E{int(smeta[-1])+1}', fontsize=16)
    
    # Mostrar la gráfica
    plt.legend(loc="upper right")
    ax_freq.legend().set_visible(False) 
    ax_freq_imp.legend().set_visible(False) 
    
    plt.subplots_adjust(hspace=0.2, wspace=0.3) 
    fig.suptitle(title, fontsize=20)
    
    