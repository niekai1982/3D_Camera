import cv2
import os
import matplotlib as mpl
mpl.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt


def generate_chessboard(img_size, chessboard_size, square_size, file_name):
    spacing = square_size
    width = img_size[0]
    height = img_size[1]
    cols = chessboard_size[0]
    rows = chessboard_size[1]
    xspacing = (width - cols * square_size) // 2
    yspacing = (height - rows * square_size) // 2
    out_1 = np.ones(shape=(img_size[1], img_size[0]), dtype=np.uint8) * 255
    for x in range(0, cols):
        for y in range(0, rows):
            if x % 2 == y % 2:
                out_1[y * spacing + yspacing:y * spacing + yspacing + spacing,
                x * spacing + xspacing:x * spacing + xspacing + spacing] = 0
                # square = SVG("rect", x=x * spacing + xspacing, y=y * spacing + yspacing, width=spacing,
                #              height=spacing, fill="black", stroke="none")
                # self.g.append(square)
    out_2 = 255 - out_1
    return out_1, out_2

if __name__ == "__main__":
    img_size = (1280, 720)
    chess_board_size = (10, 8)
    square_size = 50
    out_1, out_2 = generate_chessboard(img_size, chess_board_size, square_size, "test")
    white = np.ones_like(out_1) * 255
    black = np.zeros_like(out_1)
    cv2.imwrite("chessboard.jpg", out_1)
    cv2.imwrite("chessboard_inv.jpg", out_2)
    cv2.imwrite("white.jpg", white)
    cv2.imwrite("black.jpg", black)