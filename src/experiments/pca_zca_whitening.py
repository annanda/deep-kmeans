import cv2
import numpy as np
from scipy import linalg

# codigo da patricia com alteracoes para aplicar a imagens coloridas.
##### Variaveis globais para realizar determinadas acoes #####
showImages = False
resizeImages = False

'''Funcao responsavel por receber um array de imagens, uma opcao para apresentar imagens e dar resize nas mesmas.
Para cada imagem do array sera calculada a sua matriz relativa a imagem esbranquicada e adicionada ao array de retorno.'''
def whiten_images(images, show, resize):
    global showImages, resizeImages
    showImages = show
    resizeImages = resize
    whitened_images = []
    for img in images:
        whitened_images.append(whiten_image(img))

'''Funcao responsavel por receber uma imagem e retornar a matriz equivalente a mesma, porem esbranquicada.'''
def whiten_image(img):
    # x = loadData(img)
    img = cv2.imread(img)
    showImage(img, 'original')
    b, g, r = cv2.split(img)
    width, height = b.shape
    shaped_b = reshapeImage(b)
    shaped_g = reshapeImage(g)
    shaped_r = reshapeImage(r)

    # canal blue
    xPCAWhite_b, U_b = PCAWhitening(shaped_b)
    xZCAWhite_b = ZCAWhitening(xPCAWhite_b, U_b)

    # canal green
    xPCAWhite_g, U_g = PCAWhitening(shaped_g)
    xZCAWhite_g = ZCAWhitening(xPCAWhite_g, U_g)

    # canal red
    xPCAWhite_r, U_r = PCAWhitening(shaped_r)
    xZCAWhite_r = ZCAWhitening(xPCAWhite_r, U_r)

    final_xZCAWhite_b = shapeImageWhitened(xZCAWhite_b, width, height)
    final_xZCAWhite_g = shapeImageWhitened(xZCAWhite_g, width, height)
    final_xZCAWhite_r = shapeImageWhitened(xZCAWhite_r, width, height)

    img = cv2.merge((final_xZCAWhite_b, final_xZCAWhite_g, final_xZCAWhite_r))
    showImage(img, 'whitened ZCA')
    final_xZCAWhite_b = shapeImageWhitened(xPCAWhite_b, width, height)
    final_xZCAWhite_g = shapeImageWhitened(xPCAWhite_g, width, height)
    final_xZCAWhite_r = shapeImageWhitened(xPCAWhite_r, width, height)
    img = cv2.merge((final_xZCAWhite_b, final_xZCAWhite_g, final_xZCAWhite_r))
    showImage(img, 'whitened PCA')

    return img

'''Funcao responsavel por ler a imagem original em tons de cinza, dar resize na mesma e apresenta-la 
(se as opcoes tiverem sido escolhidas). Retorna uma matriz 2D (pois transformamos a imagem para tons 
de cinza) NxM, onde N e M sao as dimensoes da imagem apos ter sido redimensionada (ou nao).'''
def loadData(img):
    img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if resizeImages:
        img_gray = cv2.resize(img_gray, (67, 540))
    
    if showImages: 
        showImage(img_gray, 'original')
    return img_gray
    
'''Funcao responsavel por receber a matriz no formato 1 x N*M de valores float e realizar os passos do
algoritmo de branqueamento do PCA. Sao eles: calcular sigma e seus autovalores (matriz U de rotacao, etc), 
achar xRot (dados rotacionados), xHat(dados em dimensao reduzida 1) e, finalmente, computar a matriz PCA 
utilizando a formula estabelecida. Retorna a matriz PCA e U.'''
def PCAWhitening(x):
    sigma = x.dot(x.T) / x.shape[1]
    U, S, Vh = linalg.svd(sigma)

    xRot = U.T.dot(x)

    #Reduz o numero de dimensoes de 2 pra 1
    k = 1
    xRot = U[:,0:k].T.dot(x)
    xHat = U[:,0:k].dot(xRot)
    
    epsilon = 1e-5
    xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T).dot(x) #formula do PCAWhitening
    return xPCAWhite, U

'''Funcao responsavel por retornar a matriz ZCAWhitening a partir da PCAWhitening e U, atraves
da formula estabelecida (UxPCAWhite).'''
def ZCAWhitening(xPCAWhite, U):
    xZCAWhite = U.dot(xPCAWhite) #formula da ZCAWhitening
    return xZCAWhite

'''Funcao responsavel por receber a matriz 2D NxM da imagem e retornar uma nova matriz 1 x N*M, sem alteracao
dos valores da mesma. Todos os valores foram transformados para float, pois quando lidamos com int temos problema 
de overflow em algumas contas.'''
def reshapeImage(img):
    vector = img.reshape(1, img.size)
    x = vector.astype('float64')
    return x

'''Funcao responsavel por abrir uma janela do sistema com o titulo escolhido para apresentar a imagem do parametro.'''
def showImage(img, title):
    cv2.imshow(title,img)
    cv2.waitKey(0)

'''Funcao responsavel por transformar a matriz ZCAWhite que possui dimensao 1xN*M em uma nova matriz, equivalente a original
de dimensoes NxM, sem alteracao de seus valores. Apresenta a imagem final caso a opcao seja escolhida.'''
def shapeImageWhitened(xZCAWhite,width, height):
    reshaped = xZCAWhite.reshape(width, height)
    reshaped_t = reshaped
    # if showImages:
        # showImage(reshaped_t, 'whitened')
        
    return reshaped_t
    
    
if __name__ == '__main__':
    whiten_images(['../../flower.jpg'], True, False)
