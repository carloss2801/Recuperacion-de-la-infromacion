import pandas as pd #libreria que abre el excel
import numpy as np
import re #Expresiones regulares, manejo de cadenas
from nltk.corpus import stopwords #Palabras vacias, palabras que no se usaran por no tener valor para el sistema
from nltk.stem import PorterStemmer #Lematizador, algoritmo que agrupa palabras por su raiz

#Clase que sirve como fila en la tabla del vocabulario
class ItemVocabulario:
	# Constructor
	def __init__(self):
		self.termino = ""
		self.listaFrecuencias = []
		self.listaDocumentos = []
	"""def __init__(self, termino, listaFrecuencias, listaDocumentos, clasificador):
		# Instance attributes
		self.termino = termino
		self.listaFrecuencias = listaFrecuencias
		self.listaDocumentos = listaDocumentos
		self.clasificador = clasificador"""

class TablaVocabulario: 
	#constructor vacio
	def __init__(self):
		self.itemsVocabulario = []  

	#constructor con lista de ItemVocabulario
	"""def __init__(self, itemsVocabulario):
		self.itemsVocabulario = itemsVocabulario"""

class TablaAtributoClas:
	def __init__(self):
		self.documentos = []
		self.clase = []
#funcion que regresa el puntero para escribir 
#sobre el archivo llamado como el contenido del parametro nombreArchivo
def leerExcel(nombreArchivo, columnas, filas):
	try:
		excel = pd.read_excel(nombreArchivo, sheet_name=0, header = None, usecols = columnas, na_filter=False, index_col=None, nrows = filas)
		return excel
	except:
		print("Error al abrir el excel")

def generarTablaClases(excel):
	tabla = []
	for i in range(excel.shape[0]):
		aux = excel[1][i] #aux ahora contiene el atributo clasificador
		aux = re.sub(r'[^MF]','',aux)
		tabla.append(aux)
	return tabla


def generarTablaVocabulario(excel):
	tf = 0 #frecuencia del termino
	N = 0 #Numero de documentos en los que aparece el termino
	tabla = TablaVocabulario()
	filaTabla = ItemVocabulario() 
	stop_words = stopwords.words('english') #Conjunto de stopwords
	sp = PorterStemmer()#lematizador 
	for i in range(excel.shape[0]): #exel.shape[0] son las filas, exel.shape[1] las columnas
		vocabulario = [] #lista donde se guarda el vocabulario sin stopwords

		aux = excel[0][i] #obtenemos el texto
		aux = aux.lower() #Lo convertimos a minusculas
		aux = aux.replace("\n"," ") #quitamos saltos de linea
		aux = re.sub(r'[^a-zA-Z\s]',' ',aux) #quitamos signos de puntuacion y reemplazamos por espacios

		palabras = aux.split() #Lista de las palabras en aux (aun tiene stopwords)
		aux = excel[1][i] #aux ahora contiene el atributo clasificador
		aux = re.sub(r'[^MF]','',aux)

		#ciclo en el que solo guardamos las palabras que no son stopword y que 
		#tienen una longitud mayor a 2    
		for j in range(len(palabras)):
			found = 0 #variable que indica si el termino esta en el vocabulario
			filaTabla = ItemVocabulario()
			if((palabras[j] not in stop_words)):
				if(len(palabras[j])>2): 
					palabras[j] = sp.stem(palabras[j])
					for k in range(len(tabla.itemsVocabulario)): #buscamos el termino en la lista
						if(tabla.itemsVocabulario[k].termino == palabras[j]):#Ya existe el termino, actualiza datos
							found = 1
							if (i in tabla.itemsVocabulario[k].listaDocumentos):#verificamos que ya exista la pos del documento i
								for m in range(len(tabla.itemsVocabulario[k].listaDocumentos)):#encuentra la posicion en que tiene que actualizar
									if(tabla.itemsVocabulario[k].listaDocumentos[m] == i):
										pos = m
										break
								tabla.itemsVocabulario[k].listaFrecuencias[pos] += 1 #aumenta la frecuencia
							else: #el termino ya existe pero no se ha agregado este documento
								tabla.itemsVocabulario[k].listaDocumentos.append(i)
								tabla.itemsVocabulario[k].listaFrecuencias.append(1)

					if (found == 0): #no se encontró, se inserta la fila en la tabla
						filaTabla.termino = palabras[j] 
						filaTabla.listaFrecuencias.append(1) 
						filaTabla.listaDocumentos.append(i)
						tabla.itemsVocabulario.append(filaTabla)
	"""En este punto la tabla con los terminos ya está creada en la variable 'tabla'
	ahora eliminamos """
	return tabla

def generarMatrizTF(tabla, excel):
	tamanioVocabulario = len(tabla.itemsVocabulario)
	matriz = np.zeros((excel.shape[0],tamanioVocabulario))

	for i in range(len(matriz)): #recorre los documentos
		for j in range(tamanioVocabulario):#recorre el vocabulario
			if(i in tabla.itemsVocabulario[j].listaDocumentos): #verificamos que el termino se encunetra en ese documento
				for k in range(len(tabla.itemsVocabulario[j].listaDocumentos)):#buscamos la pos en el arreglo para recuperar su frecuencia
					if(tabla.itemsVocabulario[j].listaDocumentos[k] == i):
						pos = k
						break
				matriz[i][j] = tabla.itemsVocabulario[j].listaFrecuencias[pos] #insertamos
 
	"""for i in range(len(tabla.itemsVocabulario)):
		print(f"|{tabla.itemsVocabulario[i].termino}|", end = "")
	print("\nMatriz tf\n")
	print(matriz)"""
	return matriz

def generarMatrizIDF(tabla, excel):
	tamanioVocabulario = len(tabla.itemsVocabulario)
	matriz = np.zeros(tamanioVocabulario)
	N = excel.shape[0]
	for j in range(tamanioVocabulario):#recorre el vocabulario
		Nt = len(tabla.itemsVocabulario[j].listaDocumentos)
		matriz[j] = np.log(N/Nt)+1
	return matriz

def generarMatrizIDF_TF(tabla, excel):
	tf = generarMatrizTF(tabla, excel)
	idf = generarMatrizIDF(tabla, excel)
	guardarMatrizTF(tf)
	guardarMatrizIDF(idf)
	filas = len(tf)
	columnas = len(tf[0])
	for i in range(filas):
		tf[i] = tf[i] * idf 
	return tf
		
"""Funcion que crea el archivo listo para Weka tomando como base """
def generarARFF(vocabulario, tablaClase, matriz = np.zeros((0,0))):
	salida = open("salidaPrueba.arff", "w")
	salida.write("@relation proyecto\n")
	n = len(vocabulario.itemsVocabulario)
	filas = len(matriz)
	columnas = len(matriz[0])
	for i in range(n):
		salida.write(f"@attribute {vocabulario.itemsVocabulario[i].termino} numeric\n")
	salida.write("@attribute sexoClase {M,F}\n@data\n")
	for i in range(filas):
		salida.write("{")
		for j in range(columnas):
			if(matriz[i][j] != 0):
				salida.write(f"{j} {matriz[i][j]},")
		salida.write(f"{j+1} {tablaClase[i]}")
		salida.write("}\n")
	salida.close()

def guardarMatrizTF(matriz):
	salida = open("matrizTF.txt", "w")
	salida.write(np.array2string(matriz))
	salida.close()

def guardarMatrizIDF(matriz):
	salida = open("matrizIDF.txt", "w")
	salida.write(np.array2string(matriz))
	salida.close()




if (__name__ ==	"__main__"):
	excel = leerExcel("prueba.xlsx", "A:B", 500) #creación de DataFrame que contiene los datos del excel
	vocabulario = generarTablaVocabulario(excel) #creación de matriz de listas que contiene la estructura para crear matrices
	tablaClase = generarTablaClases(excel) #lista que contiene el valor de clasificacion de cada documento
	matriz = generarMatrizIDF_TF(vocabulario, excel)
	generarARFF(vocabulario, tablaClase, matriz)

	"""
	for i in range(len(vocabulario.itemsVocabulario)):
		print(f"{vocabulario.itemsVocabulario[i].termino}\t{vocabulario.itemsVocabulario[i].listaFrecuencias}\t{vocabulario.itemsVocabulario[i].listaDocumentos}\n**********************************\n")
	print("-------- Tabla de clases ---------")
	for i in range(len(clase.documentos)):
		print(f"{clase.documentos[i]}\t{clase.clase[i]}")"""

	#archivoSalida = open("salida.arff", "w")
	#salida.write("@relation proyecto\n@attribute texto string\n@attribute sexo {M,F}\n@data\n");