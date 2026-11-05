"""Análise espectral via FFT de um sinal com 3 componentes + ruído."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PARÂMETROS DE AMOSTRAGEM
fs = 2000.0                 # taxa de amostragem [Hz] (> 2*300 Hz, Nyquist)
T = 1.0                     # duração [s]
N = int(fs * T)             # número de amostras = 2000
t = np.arange(N) / fs       # vetor de tempo

# SINAL: 50 Hz (A=1,0), 120 Hz (A=0,5), 300 Hz (A=0,25)
A = {50: 1.0, 120: 0.5, 300: 0.25}
x_limpo = sum(a * np.sin(2 * np.pi * f * t) for f, a in A.items())

# RUÍDO GAUSSIANO (semente fixa para reprodutibilidade)
rng = np.random.default_rng(42)
sigma = 0.5
ruido = rng.normal(0, sigma, N)
x = x_limpo + ruido

# FFT
X = np.fft.fft(x)                       # FFT de N pontos
f = np.fft.fftfreq(N, d=1/fs)           # eixo de frequências
mag = (2.0 / N) * np.abs(X[:N//2])      # espectro unilateral (amplitude)
fpos = f[:N//2]
X0 = np.fft.fft(x_limpo)
mag0 = (2.0 / N) * np.abs(X0[:N//2])

# DETECÇÃO DE PICOS (limiar = 5x mediana do espectro ~ piso de ruído)
piso = np.median(mag)
limiar = 5 * piso
idx = [i for i in range(1, len(mag)-1)
       if mag[i] > limiar and mag[i] > mag[i-1] and mag[i] > mag[i+1]]
print(f"Resolução em frequência df = fs/N = {fs/N:.2f} Hz")
print(f"Piso de ruído (mediana) = {piso:.4f}; limiar = {limiar:.4f}")
for i in idx:
    print(f"Pico: f = {fpos[i]:7.2f} Hz | amplitude = {mag[i]:.4f}")
print(f"Potência do ruído teórica sigma^2 = {sigma**2:.3f}; medida = {np.var(ruido):.3f}")
P_sig = np.mean(x_limpo**2); P_r = np.mean(ruido**2)
print(f"SNR = {10*np.log10(P_sig/P_r):.2f} dB")

# GRÁFICOS
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 1, figsize=(7.5, 4.6))
ax[0].plot(t[:200]*1000, x_limpo[:200], label="Sinal limpo", lw=1.2)
ax[0].plot(t[:200]*1000, x[:200], label="Sinal + ruído", lw=0.8, alpha=.8)
ax[0].set_xlabel("Tempo [ms]"); ax[0].set_ylabel("Amplitude")
ax[0].set_title("Domínio do tempo (primeiros 100 ms)"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
ax[1].plot(fpos, mag, lw=0.8, color="tab:red")
for i in idx:
    ax[1].annotate(f"{fpos[i]:.0f} Hz\n{mag[i]:.2f}", (fpos[i], mag[i]),
                   textcoords="offset points", xytext=(18, -4), fontsize=8)
ax[1].axhline(limiar, ls="--", color="gray", lw=0.8, label="Limiar de detecção")
ax[1].set_xlim(0, 500); ax[1].set_xlabel("Frequência [Hz]"); ax[1].set_ylabel("Magnitude")
ax[1].set_title("Espectro de amplitude (FFT, 2000 pontos)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("espectro.png", dpi=200)

fig2, ax2 = plt.subplots(figsize=(7.5, 2.6))
ax2.plot(fpos, mag0, label="Sem ruído", lw=1.2)
ax2.plot(fpos, mag, label="Com ruído", lw=0.7, alpha=.8)
ax2.set_xlim(0, 1000); ax2.set_xlabel("Frequência [Hz]"); ax2.set_ylabel("Magnitude")
ax2.set_title("Comparação espectral até 1000 Hz (Nyquist)"); ax2.legend(); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig("comparacao.png", dpi=200)