from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from PIL import Image

from line import Line
from SlidingWindow import SlidingWindow

from sklearn.preprocessing import normalize

DType = TypeVar("DType", bound=np.generic)

LineVector = Annotated[npt.NDArray[DType], Literal[3]]


def main():
    img = Image.open("teeth_sample.png")
    img_data = np.array(img)

    minval = np.percentile(img_data, 11)
    maxval = np.percentile(img_data, 100)

    img_contrast = np.clip(img_data.copy(), minval, maxval)
    img_contrast = ((img_contrast - minval) / (maxval - minval)) * 255

    _, axes = plt.subplots(1, 2)
    axes[0].set_title("Original")
    axes[0].imshow(img_data, cmap="gray")
    axes[1].set_title("Contrasted")
    axes[1].imshow(img_contrast, cmap="gray")
    
    
    # Vertical window that slides horizonally over the image and finds the upper and
    # lower segmentation of the teeth
    firstWindow = SlidingWindow(img_contrast, 32, median=210, sigma=29, constant=50, strip=64)
    firstWindow.sampling(10, "Vertical Sliding Window")
    
    line_vector = np.polyfit(firstWindow.x_positions[:, 0], firstWindow.y_positions[:, 0], 2)
    up_low_seg_line = Line(line_vector, size=img_contrast.shape[1])
    
    # Second window, horizontal that slides vertically outlining the upper teeth
    top_window = SlidingWindow(img_contrast, 
                            num_steps=64, 
                            medians= [10, 105, 185, 275, 365, 480], 
                            sigmii=[20, 23, 23, 22, 40, 28], 
                            constant= 10, 
                            line=line_vector,
                            horiz=True, 
                            top=True, 
                            diagnose=False, 
                            diag_num=5)

    top_window.sampling(5, "Horizontal Top Sliding Window")
    
    # third window, horizontal that slides vertically outlineing the lower teeth
    bottom_window = SlidingWindow(img_contrast, 
                            num_steps=64, 
                            medians=[60, 145, 235, 395, 510], 
                            sigmii=[20, 20, 25, 65, 10], 
                            constant=20, 
                            line=line_vector,
                            horiz=True, 
                            bot=True, 
                            diagnose=False, 
                            diag_num=5)

    bottom_window.sampling(42, "Horizontal Bottom Sliding Window")

    plt.title("Top & Bottom Segmentation")
    plt.imshow(img_data, cmap="gray")
    
    plt.plot(up_low_seg_line.x_values, up_low_seg_line.y_values, color="red")    
    plt.show()
    
    plt.title("Top Teeth Segmentation")
    plt.imshow(img_data, cmap="gray")

    # plot the top teeth lines
    for i in range(len(top_window.medians) - 1):
        A = np.vstack([top_window.y_positions[:,i], np.ones(len(top_window.x_positions))] ).T
        n, c = np.linalg.lstsq(A, top_window.x_positions[:,i], rcond=None)[0]

        plt.ylim(top_window.img_height, 0)
        plt.plot(n * top_window.y_positions[:,i] + c, top_window.y_positions[:,i])#, slope=tooth_line[0])
    
    # plt.scatter(top_window.x_positions[:, i], top_window.y_positions[:, i])
        # plt.plot(top_window.x_positions[:,i], func(top_window.x_positions[:,i]))
    
    plt.plot(up_low_seg_line.x_values, up_low_seg_line.y_values, color="red")    
    plt.show()
    
    plt.title("Bottom Teeth segmentation")
    plt.imshow(img_data, cmap="gray")
    # plot the bottom teeth lines
    for i in range(len(bottom_window.medians)):
        A = np.vstack([bottom_window.y_positions[:,i], np.ones(len(bottom_window.x_positions))] ).T
        n, c = np.linalg.lstsq(A, bottom_window.x_positions[:,i], rcond=None)[0]

        plt.ylim(bottom_window.img_height, 0)
        plt.plot(n * bottom_window.y_positions[:,i] + c, bottom_window.y_positions[:,i])#, slope=tooth_line[0])
        
        # plt.scatter(top_window.x_positions[:, i], top_window.y_positions[:, i])

    plt.plot(up_low_seg_line.x_values, up_low_seg_line.y_values, color="red")      
    plt.show()
    
    plt.title("Final segmentation")
    plt.imshow(img_data, cmap="gray")
    plt.plot(up_low_seg_line.x_values, up_low_seg_line.y_values, color="red")
    
    for i in range(len(bottom_window.medians)):
        A_top = np.vstack([top_window.y_positions[:,i], np.ones(len(top_window.x_positions))] ).T
        A_bot = np.vstack([bottom_window.y_positions[:,i], np.ones(len(bottom_window.x_positions))] ).T
        
        n_top, c_top = np.linalg.lstsq(A_top, top_window.x_positions[:,i], rcond=None)[0]
        n_bot, c_bot = np.linalg.lstsq(A_bot, bottom_window.x_positions[:,i], rcond=None)[0]

        plt.ylim(bottom_window.img_height, 0)
        
        plt.plot(n_top * top_window.y_positions[:,i] + c_top, top_window.y_positions[:,i])#, slope=tooth_line[0])
        plt.plot(n_bot * bottom_window.y_positions[:,i] + c_bot, bottom_window.y_positions[:,i])#, slope=tooth_line[0])
        
    plt.show()

main()