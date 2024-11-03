import numpy as np
from typing import Tuple, List

class AlgoritmoHungaro:
    """
    Implementação do Algoritmo Húngaro para resolver problemas de atribuição.
    O algoritmo encontra a atribuição ótima (mínima ou máxima) em uma matriz de custos.
    """
    
    def __init__(self, matriz_custos: np.ndarray):
        """
        Inicializa o solver com a matriz de custos.
        
        Parâmetros
        ----------
        matriz_custos : np.ndarray
            Matriz quadrada com os custos de cada atribuição
        """
        self.matriz_original = np.array(matriz_custos, dtype=float)
        self.matriz_custos = self.matriz_original.copy()
        self.dimensao = matriz_custos.shape[0]
        self.posicoes_otimas = None
        self.custo_total = None
        self.matriz_resposta = None

    def _min_zero_linha(self, matriz_zeros: np.ndarray, zeros_marcados: List) -> None:
        """
        Encontra a linha com menor número de zeros e marca um deles.
        
        Parâmetros
        ----------
        matriz_zeros : np.ndarray
            Matriz booleana indicando posição dos zeros
        zeros_marcados : List
            Lista para armazenar as posições dos zeros marcados
        """
        min_linha = [99999, -1]

        for num_linha in range(matriz_zeros.shape[0]): 
            if (np.sum(matriz_zeros[num_linha] == True) > 0 and 
                min_linha[0] > np.sum(matriz_zeros[num_linha] == True)):
                min_linha = [np.sum(matriz_zeros[num_linha] == True), num_linha]

        indice_zero = np.where(matriz_zeros[min_linha[1]] == True)[0][0]
        zeros_marcados.append((min_linha[1], indice_zero))
        matriz_zeros[min_linha[1], :] = False
        matriz_zeros[:, indice_zero] = False

    def _marcar_matriz(self, matriz: np.ndarray) -> Tuple[List, List, List]:
        """
        Encontra e marca os zeros na matriz para construir a solução.
        
        Parâmetros
        ----------
        matriz : np.ndarray
            Matriz de trabalho atual
            
        Retorna
        -------
        Tuple[List, List, List]
            zeros_marcados, linhas_marcadas, colunas_marcadas
        """
        matriz_atual = matriz
        matriz_bool_zeros = (matriz_atual == 0)
        copia_matriz_bool_zeros = matriz_bool_zeros.copy()

        zeros_marcados = []
        while (True in copia_matriz_bool_zeros):
            self._min_zero_linha(copia_matriz_bool_zeros, zeros_marcados)
        
        linhas_zeros_marcados = []
        colunas_zeros_marcados = []
        for i in range(len(zeros_marcados)):
            linhas_zeros_marcados.append(zeros_marcados[i][0])
            colunas_zeros_marcados.append(zeros_marcados[i][1])

        linhas_nao_marcadas = list(set(range(matriz_atual.shape[0])) - set(linhas_zeros_marcados))
        
        colunas_marcadas = []
        verificar_mudanca = True
        while verificar_mudanca:
            verificar_mudanca = False
            for i in range(len(linhas_nao_marcadas)):
                array_linha = matriz_bool_zeros[linhas_nao_marcadas[i], :]
                for j in range(array_linha.shape[0]):
                    if array_linha[j] == True and j not in colunas_marcadas:
                        colunas_marcadas.append(j)
                        verificar_mudanca = True

            for num_linha, num_coluna in zeros_marcados:
                if (num_linha not in linhas_nao_marcadas and 
                    num_coluna in colunas_marcadas):
                    linhas_nao_marcadas.append(num_linha)
                    verificar_mudanca = True
                    
        linhas_marcadas = list(set(range(matriz.shape[0])) - set(linhas_nao_marcadas))
        return zeros_marcados, linhas_marcadas, colunas_marcadas

    def _ajustar_matriz(self, matriz: np.ndarray, linhas_cobertas: List, 
                       colunas_cobertas: List) -> np.ndarray:
        """
        Ajusta a matriz para a próxima iteração do algoritmo.
        
        Parâmetros
        ----------
        matriz : np.ndarray
            Matriz de trabalho atual
        linhas_cobertas : List
            Lista de índices das linhas cobertas
        colunas_cobertas : List
            Lista de índices das colunas cobertas
            
        Retorna
        -------
        np.ndarray
            Matriz ajustada
        """
        matriz_atual = matriz
        elementos_nao_zero = []

        for linha in range(len(matriz_atual)):
            if linha not in linhas_cobertas:
                for i in range(len(matriz_atual[linha])):
                    if i not in colunas_cobertas:
                        elementos_nao_zero.append(matriz_atual[linha][i])
                        
        numero_min = min(elementos_nao_zero)

        for linha in range(len(matriz_atual)):
            if linha not in linhas_cobertas:
                for i in range(len(matriz_atual[linha])):
                    if i not in colunas_cobertas:
                        matriz_atual[linha, i] = matriz_atual[linha, i] - numero_min
                        
        for linha in range(len(linhas_cobertas)):  
            for col in range(len(colunas_cobertas)):
                matriz_atual[linhas_cobertas[linha], colunas_cobertas[col]] += numero_min
                
        return matriz_atual

    def resolver(self, maximizar: bool = False) -> Tuple[float, np.ndarray]:
        """
        Resolve o problema de atribuição.
        
        Parâmetros
        ----------
        maximizar : bool, opcional (default = False)
            Se True, encontra a atribuição de valor máximo.
            Se False, encontra a atribuição de valor mínimo.
            
        Retorna
        -------
        Tuple[float, np.ndarray]
            custo_total: Valor total da atribuição ótima
            matriz_resposta: Matriz com apenas os elementos escolhidos
        """
        if maximizar:
            valor_max = np.max(self.matriz_custos)
            self.matriz_custos = valor_max - self.matriz_custos
        
        matriz_trabalho = self.matriz_custos.copy()
        
        # Passo 1: Redução da matriz
        for num_linha in range(self.dimensao): 
            matriz_trabalho[num_linha] = matriz_trabalho[num_linha] - np.min(matriz_trabalho[num_linha])
        
        for num_coluna in range(self.dimensao): 
            matriz_trabalho[:,num_coluna] = matriz_trabalho[:,num_coluna] - np.min(matriz_trabalho[:,num_coluna])
        
        # Processo iterativo principal
        contador_zeros = 0
        while contador_zeros < self.dimensao:
            pos_resposta, linhas_marcadas, colunas_marcadas = self._marcar_matriz(matriz_trabalho)
            contador_zeros = len(linhas_marcadas) + len(colunas_marcadas)

            if contador_zeros < self.dimensao:
                matriz_trabalho = self._ajustar_matriz(matriz_trabalho, linhas_marcadas, colunas_marcadas)

        # Calcula e armazena os resultados
        self.posicoes_otimas = pos_resposta
        self.matriz_resposta = np.zeros((self.dimensao, self.dimensao))
        self.custo_total = 0
        
        for i in range(len(self.posicoes_otimas)):
            linha, coluna = self.posicoes_otimas[i]
            if maximizar:
                self.matriz_resposta[linha, coluna] = self.matriz_original[linha, coluna]
                self.custo_total += self.matriz_original[linha, coluna]
            else:
                self.matriz_resposta[linha, coluna] = self.matriz_original[linha, coluna]
                self.custo_total += self.matriz_original[linha, coluna]
                
        return self.custo_total, self.matriz_resposta

    def obter_atribuicoes(self) -> List[Tuple[int, int, float]]:
        """
        Retorna a lista de atribuições na forma (linha, coluna, custo).
        
        Retorna
        -------
        List[Tuple[int, int, float]]
            Lista de tuplas (linha, coluna, custo) representando cada atribuição
        """
        if self.posicoes_otimas is None:
            raise ValueError("Execute o método 'resolver()' primeiro.")
            
        atribuicoes = []
        for linha, coluna in self.posicoes_otimas:
            custo = self.matriz_original[linha, coluna]
            atribuicoes.append((linha, coluna, custo))
            
        return sorted(atribuicoes)

    def imprimir_solucao(self) -> None:
        """
        Imprime a solução de forma formatada.
        """
        if self.posicoes_otimas is None:
            print("Execute o método 'resolver()' primeiro.")
            return
            
        print(f"\nCusto total da atribuição: {self.custo_total:.2f}")
        print("\nAtribuições:")
        for linha, coluna, custo in self.obter_atribuicoes():
            print(f"Agente {linha} -> Tarefa {coluna} (custo: {custo:.2f})")
            
        print("\nMatriz de atribuição (mostra apenas elementos escolhidos):")
        print(self.matriz_resposta)