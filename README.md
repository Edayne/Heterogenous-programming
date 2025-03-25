# Rendu projet et TP de Heterogenous Programming

- Auteurs : Amandine HENRY, Saïf-Eddine KEHILI


# Décisions prises
On constate sur cet exemple que l'éxécution d'un filtre bilinéaire sur une image 512*512 est 2 fois plus rapide après une parallélisation GPU (144 ms) que sur une éxécution séquentielle CPU (334 ms).

Le calcul du temps d'éxécution dans le cas de la parallélisation est biaisé dû à une utilisation de cudaEventCreate() qui ne calcule que le temps d'éxécution sur le GPU à proprement parler.

Il est à noter que le test CPU a été effectué sur ma machine personelle, le test GPU a quant à lui été effectué sur une machine munie d'une RTX A2000.


# Préparation

```sh
git clone https://github.com/Edayne/Heterogenous-programming
cd Heterogenous-programming
``̀

# Execution CPU

```sh
gcc main.c -o main -lm
./main lena512.bmp output.bmp
```

# Execution GPU

```sh
nvcc main.cu -o main -lm
./main lena512.bmp output.bmp
```
