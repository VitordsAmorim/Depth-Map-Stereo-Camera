"""
Módulo de Processamento Estéreo e Reconstrução 3D

Este módulo implementa diversas funções para processar imagens estereoscópicas,
incluindo a calibração de câmeras, detecção e correspondência de features, cálculo
do mapa de disparidade, reconstrução de cenas 3D e análise da geometria epipolar.

Bibliotecas utilizadas:
    - NumPy: operações matriciais e manipulação de arrays.
    - OpenCV (cv2): processamento de imagens, detecção de features, cálculo de disparidade e
      geometria epipolar.
    - Matplotlib: visualização de imagens e gráficos.
"""

import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt


def camera_parameters_empresa():
    """
    Retorna os parâmetros internos (matrizes de calibração) das câmeras esquerda e direita,
    conforme definidos pela empresa.

    A função define os parâmetros individuais para a câmera esquerda e direita e retorna
    as respectivas matrizes de calibração.

    Returns:
        tuple: Um par de matrizes numpy (kl, kr) correspondentes à calibração da câmera
               esquerda e direita, respectivamente.
    """
    # Parâmetros da câmera esquerda
    fx = 690.1220738240332
    fy = 690.1220738240332
    cx = 641.5391125464108
    cy = 336.57310036889279

    kl = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=float)

    # Parâmetros da câmera direita
    fx = 693.9171821166054
    fy = 693.9171821166054
    cx = 636.4770692826232
    cy = 368.3221311054167

    kr = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=float)

    return kl, kr


def show_matches(img_l, img_r, kp_l, kp_r, matches):
    """
    Exibe a imagem resultante com as correspondências (matches) entre keypoints
    das imagens esquerda e direita.

    Utiliza a função drawMatches do OpenCV para desenhar as linhas conectando
    os keypoints correspondentes, converte a imagem para RGB e a exibe com Matplotlib.

    Args:
        img_l (numpy.ndarray): Imagem da câmera esquerda.
        img_r (numpy.ndarray): Imagem da câmera direita.
        kp_l (list): Lista de keypoints da imagem esquerda.
        kp_r (list): Lista de keypoints da imagem direita.
        matches (list): Lista de correspondências entre os keypoints.
    """
    img3 = cv.drawMatches(
        img_l, kp_l, img_r, kp_r, matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Converte de BGR para RGB para exibição correta com Matplotlib
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()
    cv2.imwrite('output/matches_output.png', cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))


def remove_outliers(matches, kp1, kp2, img):
    """
    Remove correspondências (matches) consideradas outliers com base na diferença
    vertical dos pontos (eixo y) dos keypoints.

    Se a diferença vertical entre os pontos correspondentes for maior que 10% da altura
    da imagem, a correspondência é descartada.

    Args:
        matches (list): Lista de correspondências (matches) entre keypoints.
        kp1 (list): Lista de keypoints da primeira imagem.
        kp2 (list): Lista de keypoints da segunda imagem.
        img (numpy.ndarray): Imagem utilizada para determinar a escala (altura).

    Returns:
        list: Lista de correspondências filtradas, sem outliers.
    """
    output_matches = []
    for m in matches:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        if abs(p1[1] - p2[1]) < (0.1 * img.shape[0]):
            output_matches.append(m)
    return output_matches


def detect_and_match_features(img_l, img_r):
    """
    Detecta e emparelha features entre duas imagens utilizando o detector ORB
    e o matcher BFMatcher.

    A função converte as imagens para escala de cinza, detecta os keypoints e
    calcula os descritores, realiza a correspondência entre os descritores e
    filtra outliers com base na diferença vertical dos pontos.

    Args:
        img_l (numpy.ndarray): Imagem da câmera esquerda.
        img_r (numpy.ndarray): Imagem da câmera direita.

    Returns:
        tuple: Contendo:
            - kp_l (list): Lista de keypoints da imagem esquerda.
            - kp_r (list): Lista de keypoints da imagem direita.
            - matches (list): Lista de correspondências (matches) filtradas.
            - pixels_l (numpy.ndarray): Array de coordenadas dos keypoints na imagem esquerda.
            - pixels_r (numpy.ndarray): Array de coordenadas dos keypoints na imagem direita.
    """
    # Inicializa o detector ORB
    orb = cv2.ORB_create()

    img_l_gray = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
    img_r_gray = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

    # Detecta keypoints e calcula descritores para ambas as imagens
    kp_l, des_l = orb.detectAndCompute(img_l_gray, None)
    kp_r, des_r = orb.detectAndCompute(img_r_gray, None)

    # Cria o objeto BFMatcher com a distância Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Realiza a correspondência entre os descritores
    matches = bf.match(des_l, des_r)

    # Filtra os outliers com base na diferença vertical dos keypoints
    matches = remove_outliers(matches, kp_l, kp_r, img_l_gray)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extrai as coordenadas dos pontos correspondentes
    pixels_l = [kp_l[mat.queryIdx].pt for mat in matches]
    pixels_r = [kp_r[mat.trainIdx].pt for mat in matches]

    # Converte as listas em arrays NumPy
    pixels_l = np.array(pixels_l, dtype=float)
    pixels_r = np.array(pixels_r, dtype=float)

    return kp_l, kp_r, matches, pixels_l, pixels_r


def compute_disparity(img_l, img_r):
    """
    Calcula o mapa de disparidade entre duas imagens estereoscópicas utilizando o
    algoritmo StereoBM do OpenCV.

    As imagens são convertidas para escala de cinza, e a disparidade é calculada,
    normalizada para exibição e exibida com Matplotlib.

    Args:
        img_l (numpy.ndarray): Imagem da câmera esquerda.
        img_r (numpy.ndarray): Imagem da câmera direita.

    Returns:
        numpy.ndarray: Mapa de disparidade normalizado (imagem em tons de cinza).
    """
    img_l_gray = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
    img_r_gray = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

    # Cria o objeto StereoBM para cálculo da disparidade
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=15)
    disparity = stereo.compute(img_l_gray, img_r_gray)

    # Normaliza o mapa de disparidade para exibição
    mini = disparity.min()
    max_val = disparity.max()
    disparity_norm = np.uint8(255 * (disparity - mini) / (max_val - mini))

    plt.imshow(disparity_norm, cmap='gray')
    plt.show()
    cv2.imwrite('output/depth_map.png', cv2.cvtColor(disparity_norm, cv2.COLOR_RGB2BGR))

    #cv2.imwrite('depth_map.png', cv2.cvtColor(disparity_norm, cv2.COLOR_RGB2BGR))

    return disparity_norm


def epipolar_geometry(i1, i2):
    """
    Calcula a geometria epipolar entre duas imagens utilizando SIFT para detecção
    de features e FLANN para correspondência.

    A função:
        - Lê as imagens em escala de cinza.
        - Detecta keypoints e extrai descritores com SIFT.
        - Realiza correspondência com FLANN e aplica o teste da razão (Lowe).
        - Calcula a matriz fundamental (F) e filtra os inliers.
        - Desenha as linhas epipolares em ambas as imagens para visualização.

    Args:
        i1 (str): Caminho para a primeira imagem.
        i2 (str): Caminho para a segunda imagem.

    Returns:
        tuple: Contendo a matriz fundamental (F), os pontos inliers da primeira imagem (pts1)
               e os pontos inliers da segunda imagem (pts2).
    """
    img_l = cv.imread(i1, 0)
    img_r = cv.imread(i2, 0)

    sift = cv.SIFT_create()

    # Detecta keypoints e calcula descritores
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_r, None)

    # Parâmetros para FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []

    # Aplica o teste de razão conforme Lowe
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # Seleciona apenas os inliers
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Calcula e desenha as linhas epipolares nas duas imagens
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img_l, img_r, lines1, pts1, pts2)

    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img_r, img_l, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
    


    return F, pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Desenha linhas epipolares e marca os pontos correspondentes nas imagens.

    Converte as imagens (originais em escala de cinza) para BGR, desenha cada linha epipolar
    e marca os pontos com círculos coloridos, utilizando uma cor aleatória para cada par.

    Args:
        img1 (numpy.ndarray): Imagem onde as linhas serão desenhadas (em escala de cinza).
        img2 (numpy.ndarray): Imagem onde os pontos serão marcados (em escala de cinza).
        lines (numpy.ndarray): Array com os coeficientes das linhas epipolares.
        pts1 (array-like): Coordenadas dos pontos na primeira imagem.
        pts2 (array-like): Coordenadas dos pontos na segunda imagem.

    Returns:
        tuple: Duas imagens (img1, img2) com as linhas epipolares e pontos desenhados.
    """
    r, c = img1.shape
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        # Gera uma cor aleatória
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Calcula os pontos de interseção da linha com as bordas da imagem
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv.circle(img1_color, tuple(np.int32(pt1)), 5, color, -1)
        img2_color = cv.circle(img2_color, tuple(np.int32(pt2)), 5, color, -1)

    return img1_color, img2_color


def reprojection_error(ul, kl, kr, e):
    """
    Calcula e imprime o erro de reprojeção utilizando os parâmetros de calibração das câmeras
    e a matriz essencial (e).

    A função realiza o cálculo da pseudo-inversa das matrizes de calibração para auxiliar no
    cálculo do erro. Atualmente, a implementação imprime os valores intermediários.

    Args:
        ul (array-like): Coordenadas dos pontos (por exemplo, keypoints 2D).
        kl (numpy.ndarray): Matriz de calibração da câmera esquerda.
        kr (numpy.ndarray): Matriz de calibração da câmera direita.
        e (numpy.ndarray): Matriz essencial.

    Returns:
        None.
    """
    ul = np.asarray(ul)
    ul_t = ul.transpose()
    print("Matriz de calibração esquerda (kl):")
    print(kl)
    kl_inv = np.linalg.pinv(kl)
    print("Pseudo-inversa de kl:")
    print(kl_inv)
    kl_inv_transpose = kl_inv.T
    print("Transposta da pseudo-inversa de kl:")
    print(kl_inv_transpose)
    print("Matriz de calibração direita (kr):")
    print(kr)
    kr_inv = np.linalg.pinv(kr)
    print("Pseudo-inversa de kr:")
    print(kr_inv)


def main():
    """
    Função principal que integra o pipeline de processamento estereoscópico.

    Para cada par de imagens listado em 'reference.txt', a função:
        - Lê os nomes das imagens e carrega as imagens.
        - Detecta e emparelha features utilizando ORB.
        - Exibe as correspondências entre os keypoints.
        - Calcula o mapa de disparidade entre as imagens.
        - Realiza a normalização dos pontos para o cálculo da matriz essencial.
        - Calcula a matriz essencial e recupera a pose (rotação e translação) entre as imagens.
        - (Opcional) Pode reconstruir a cena 3D ou calcular o erro de reprojeção.

    A execução é interrompida após o processamento do primeiro par de imagens.
    """
    count, n = 0, 0
    arquivo = open('reference.txt', 'r')
    for linha in arquivo:
        # Lê o par de nomes de arquivos de imagem
        i1, i2 = linha.strip().split()
        img_l = cv2.imread(i1)
        img_r = cv2.imread(i2)

        # Detecta e emparelha features entre as imagens
        kp_l, kp_r, matches, pixels_l, pixels_r = detect_and_match_features(img_l, img_r)
        show_matches(img_l, img_r, kp_l, kp_r, matches)

        # Calcula o mapa de disparidade
        disparity = compute_disparity(img_l, img_r)

        # Normaliza pontos para o cálculo da matriz essencial
        kl, kr = camera_parameters_empresa()
        pts_l_norm = cv2.undistortPoints(np.expand_dims(pixels_l, axis=1), cameraMatrix=kl, distCoeffs=None)
        pts_r_norm = cv2.undistortPoints(np.expand_dims(pixels_r, axis=1), cameraMatrix=kr, distCoeffs=None)

        # Utiliza a matriz identidade, já que os parâmetros foram normalizados
        k = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)

        # Calcula a matriz essencial utilizando RANSAC
        e, _ = cv2.findEssentialMat(pts_l_norm, pts_r_norm, k, method=cv2.RANSAC)
        points, R, t, _ = cv2.recoverPose(e, pts_l_norm, pts_r_norm, cameraMatrix=k)

        #reconstruction_3d(R, t, kr, img_r)
        # Exibe os resultados
        print("Matriz Essencial:")
        print(e)
        print("Matriz de Rotação:")
        print(R)
        print("Matriz de Translação:")
        print(t)

        # Processa apenas o primeiro par de imagens
        
        break

    n = n + 1
    arquivo.close()


if __name__ == "__main__":
    main()