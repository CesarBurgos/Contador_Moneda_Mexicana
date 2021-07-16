import numpy as np
import cv2

''' ___ Función: OrdenarPuntos ___
    Esta función retornará los puntos o coordenadas del área a considerar
    como el área de trabajo.
'''
def OrdenarPuntos(puntos):
    n_puntos = np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()

    # ____ Ordenamiento de coordenadas ____
    y_order = sorted(n_puntos, key = lambda CoordY: CoordY[1])
    
    x_Superior = y_order[0:2]
    x_Superior = sorted(x_Superior, key = lambda CoordX_Sup: CoordX_Sup[0])
    
    x_Inferior = y_order[2:4]
    x_Inferior = sorted(x_Inferior, key = lambda CoordX_Inf: CoordX_Inf[0])
    
    return [x_Superior[0],x_Superior[1],x_Inferior[0],x_Inferior[1]]


''' ___Función: area_trabajo_alineamiento ___
    Esta función reaaliza lo siguiente:
    
    1° Detección del fondo... esto con el objetivo de enfocarse
       en detectar el fondo de la ímagen (área de teabajo)

    2° Determina si el contorno detectado es el área de trabajo
       deseado, esto lo lleva a cabo determinado el perímetro
       del contorno

    3° Crea una nueva imagen si la perspectiva de esta misma es
       diferente a lo deseado
'''
def area_trabajo_alineamiento(Img,w,h):
    img_alineada = None

    # ==== 1° Detectando al fondo como nuestro objeto deseado...
    #__ Conversión a Grises
    Img2 = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    #__ Umbralizando la imagen
    _, Img2 = cv2.threshold(Img2, 100, 255, cv2.THRESH_OTSU)

    #__ Detección de contornos
    Img2 = cv2.findContours(Img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Ordenamiento de los puntos obtenidos
    Img2 = sorted(Img2, key = cv2.contourArea, reverse = True)[0:1]

    # ==== 2° Detectando si el contorno encontrado es el área de trabajo deseada...
    
    # ____ Extracción de los contornos ____ 
    for contorno in Img2:
            epsilon = 0.1 * cv2.arcLength(contorno, True)
            aprox = cv2.approxPolyDP(contorno, epsilon, True)

            if len(aprox) == 4:
                # ==== 3° Modificación de la imágen ante cambios de perspectiva
                
                # Retornando los Puntos del área a considerar como área de trabajo 
                Puntos_Area_Identificada = np.float32(OrdenarPuntos(aprox))

                # Puntos del Área a considerar... (Tamaño A6, con medidas en pixeles)
                Puntos_Area_Deseada = np.float32([[0,0], [w,0], [0,h], [w,h]])

                # ___ Perspectiva - Cambio de perspectiva la imágen capturada ___
                M = cv2.getPerspectiveTransform(Puntos_Area_Identificada,Puntos_Area_Deseada)
                img_alineada = cv2.warpPerspective(Img, M, (w,h))
    return img_alineada
    

# ____ DECLARACIÓN DE VARIABLES ____
# Variable que suma al área en Pixeles de la monedas, con el objetivo de tener un rango más amplio ante posibles errores de áreas
Rango_de_Error = 200
# Fuente para textos dibujados en la captura de la cámara
font = cv2.FONT_HERSHEY_SIMPLEX
#__ Creando una matriz de unos (Elemento Estructurante), entero sin signo... 
EE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,1)) #np.ones((3,5), np.uint8)

# ____ EJECUCIÓN PRINCIPAL ____
''' ____ Lectura de frames obtenidos por la cámara: 
    0 - Cámara de la PC
    1 - DroidCamApp
    2 - Cámara Externa
'''
Cam = cv2.VideoCapture(0)

# Verificando si se encuentra activada la cámara...
if not Cam.isOpened():
    print('No hay camaras...')
else:
    #__ Bandera que indicará que estan procesando frames de la cámara
    CapturaFrames = True
    while CapturaFrames:
        CapturaFrames, Frame = Cam.read()

        # Si ya no hay otro frame por leer romper ciclo...
        if not CapturaFrames:
            break

        # Visualizando los frames que obtiene la cámara
        cv2.imshow('- Camara -', Frame)

        # w - Resolución de cámara 
        # h - El valor obtenido
        AreaTrabajo = area_trabajo_alineamiento(Frame, 480, 640)
        
        # Una vez.. preparada el área de trabajo, corresponde a la identificación de las monedas...
        if AreaTrabajo is not None:
            puntos = []

            #__  Calcular brillo
            h = cv2.cvtColor(AreaTrabajo, cv2.COLOR_BGR2HSV)
            brillo = h[:,:,2].mean()
           
            # Copiando área de trabajo para mostrar resultados finales...
            AreaTrabajoOrig = AreaTrabajo.copy()

            #__ Escala de grises 
            AreaTrabajo2 = cv2.cvtColor(AreaTrabajo, cv2.COLOR_BGR2GRAY)

            # Dependiendo de la intensidad del brillo obtenida este subirá su intensidad hasta 190 (promedio)
            if brillo <=190  and brillo >= 150:
                # Subiendo brillo
                AreaTrabajo2 += int(190 - brillo)
            
            #__ Aplicar suavizado Gaussiano - Ayuda a eliminar posible ruído en la imagen
            AreaTrabajo2 = cv2.GaussianBlur(AreaTrabajo2, (3,5), 0)
            
            #__ Umbralización
            # Se coloca de NEGRO el fondo, y las monedas en Blanco, ya que es el objeto a considerar (BINARIZACIÓN INVERTIDA)
            _, AreaTrabajo2 = cv2.threshold(AreaTrabajo2, 150, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

            #___ Morfología de Cierre... Eliminación del ruido dentro del objeto...
            AreaTrabajo2 = cv2.morphologyEx(AreaTrabajo2, cv2.MORPH_CLOSE, EE, iterations = 2)

            #__ Detección de contornos
            AreaTrabajo2 = cv2.findContours(AreaTrabajo2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            # Para almacenamiento del valor de las monedas, detectadas...
            Moneda1, Moneda2, Moneda5, Moneda10, Moneda50C = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Procesando los contornos (Monedas) obtenidas en el área de trabajo
            for contorno in AreaTrabajo2:

                # 1° - Determinando el área de las monedas...
                # (Consultar: CalculoAreaTrabajo_y_AreaDeMonedas.xls)
                area = cv2.contourArea(contorno)
                if area != 0:
                    # 2° - Detección de momentos en contorno

                    Momentos = cv2.moments(contorno)
                    
                    # __ Derterminando el centroide el contorno detectado (Moneda)
                    x = int(Momentos['m10'] / Momentos['m00'])
                    y = int(Momentos['m01'] / Momentos['m00'])
                    
                    '''
                    ____ 3° - DETERMINANDO EL VALOR DE LA MONEDA (OBJETO IDENTIFICADO) ____
                        (Consultar: CalculoAreaTrabajo_y_AreaDeMonedas.xls)
                        
                        De acuerdo a la siguiente tabla son obtenidas las áreas de las monedas a detectar...
                        
                        Valor de la moneda  | Medida(mm) |  Radio(mm) |  Área(mm^2) |  Area (px)
                        Moneda de 1 peso    |     21     |     10.5   |   346.4     |  7576.64
                        Moneda de 2 pesos   |     23     |     11.5   |   415.5     |  9088.53
                        Moneda de 5 pesos   |     25     |     12.5   |   490.9     |  10737.87
                        Moneda de 10 pesos  |     28     |      14    |   615.8     |  13469.58
                    '''
                    if area < (13469+Rango_de_Error) and area > (10737+Rango_de_Error):
                        # Por la medida de su área en pixeles puede ser la Moneda de 10 pesos
                        Moneda10 += 10.0
                        cv2.putText(AreaTrabajoOrig,'$10.00',(x,y),font,0.6, (0,0,0),2)
                        cv2.circle(AreaTrabajoOrig,(x,y),5,(0,255,0),-1)
                    elif area < (10737+Rango_de_Error) and area > (9088+Rango_de_Error):
                        # Por la medida de su área en pixeles puede ser la Moneda de 5 pesos
                        Moneda5 += 5.0
                        cv2.putText(AreaTrabajoOrig,'$5.00',(x,y),font,0.6, (0,0,0),2)
                        cv2.circle(AreaTrabajoOrig,(x,y),5,(0,255,0),-1)
                    elif area < (9088+200) and area > (7576+200):
                        # Por la medida de su área en pixeles puede ser la Moneda de 2 pesos
                        Moneda2 += 2.0
                        cv2.putText(AreaTrabajoOrig,'$2.00',(x,y),font,0.6, (0,0,0),2)
                        cv2.circle(AreaTrabajoOrig,(x,y),5,(0,255,0),-1)
                    elif area < (7576+200) and area > (4965+200):
                        # Por la medida de su área en pixeles puede ser la Moneda de 1 peso
                        Moneda1 += 1.0
                        cv2.putText(AreaTrabajoOrig,'$1.00',(x,y),font,0.6, (0,0,0),2)
                        cv2.circle(AreaTrabajoOrig,(x,y),5,(0,255,0),-1)
                    elif area < (4965+200) and area > (2365+200):
                        # Por la medida de su área en pixeles puede ser la Moneda de 50 centavos
                        Moneda50C += 0.5
                        cv2.putText(AreaTrabajoOrig,'$0.50',(x,y),font,0.6, (0,0,0),2)
                        cv2.circle(AreaTrabajoOrig,(x,y),5,(0,255,0),-1)

            # Cantidad de dinero detectado por la cámara...
            Dinero = Moneda10 + Moneda5 + Moneda2 + Moneda1 + Moneda50C
            cv2.putText(AreaTrabajoOrig, f' Dinero: $ {Dinero}0 MXN' ,(0,20),font,0.6, (0,0,0),2)

            Resultados = cv2.hconcat([AreaTrabajo, AreaTrabajoOrig])
            cv2.imshow(f'--- Deteccion de Monedas --- ',Resultados)
            
        #__ Finaliza la captura de la imágenes por la cámara al presionar la tecla X
        if cv2.waitKey(1) == ord('x') or cv2.waitKey(1) == ord('X'):
            print('Finalizando captura de imagen de la Cámara...')
            break

#__ Desactivando la cámara
Cam.release()

#__ Destruyendo todas las posibles ventanas emergentes de cv2
cv2.destroyAllWindows()