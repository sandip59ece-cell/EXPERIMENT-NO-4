import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXPERIMENT 4: UNIFORM QUANTIZATION AND PCM
# ============================================================

# ------------------------------------------------------------
# 1. Generate a sinusoidal signal
# ------------------------------------------------------------

fs = 10000          # Sampling frequency
f = 100             # Signal frequency
duration = 0.02     # seconds

t = np.arange(0, duration, 1/fs)

# Sinusoidal signal normalized to [-1, 1]
x = np.sin(2 * np.pi * f * t)

# Make sure signal is within [-1, 1]
x = np.clip(x, -1, 1)


# ------------------------------------------------------------
# 2. Uniform PCM Quantizer
# ------------------------------------------------------------

def uniform_pcm_quantizer(x, n_bits):
    """
    Uniform PCM quantizer from first principles.

    Input:
        x       : input samples in [-1, 1]
        n_bits  : number of quantization bits

    Returns:
        xq          : quantized samples
        indices     : quantizer indices
        codewords   : binary PCM words
        mse         : mean squared error
        sqnr        : measured SQNR
        theoretical : theoretical full-scale sine SQNR
    """

    # Number of quantization levels
    L = 2 ** n_bits

    # Quantization step size
    Delta = 2 / L

    # --------------------------------------------------------
    # Quantizer index
    # --------------------------------------------------------

    # Map [-1, 1] to [0, L)
    indices = np.floor((x + 1) / Delta).astype(int)

    # Prevent index L when x = +1
    indices = np.clip(indices, 0, L - 1)

    # --------------------------------------------------------
    # Reconstruction / quantized value
    # Mid-rise uniform quantizer
    # --------------------------------------------------------

    xq = -1 + (indices + 0.5) * Delta

    # --------------------------------------------------------
    # Generate binary PCM codewords
    # --------------------------------------------------------

    codewords = [format(i, f'0{n_bits}b') for i in indices]

    # --------------------------------------------------------
    # Quantization error
    # --------------------------------------------------------

    error = x - xq

    # Mean Squared Error
    mse = np.mean(error ** 2)

    # Signal power
    signal_power = np.mean(x ** 2)

    # SQNR
    sqnr = 10 * np.log10(signal_power / mse)

    # Theoretical full-scale sinusoid approximation
    theoretical = 6.02 * n_bits + 1.76

    return xq, indices, codewords, mse, sqnr, theoretical


# ------------------------------------------------------------
# 3. Run for 2 to 8 bits
# ------------------------------------------------------------

results = []

quantized_data = {}

for n in range(2, 9):

    xq, indices, codewords, mse, sqnr, theoretical = \
        uniform_pcm_quantizer(x, n)

    quantized_data[n] = {
        "quantized": xq,
        "indices": indices,
        "codewords": codewords
    }

    results.append([
        n,
        2**n,
        2/(2**n),
        mse,
        sqnr,
        theoretical,
        sqnr - theoretical
    ])


# ------------------------------------------------------------
# 4. Display results
# ------------------------------------------------------------

print("=" * 85)
print("UNIFORM PCM QUANTIZATION RESULTS")
print("=" * 85)

print(f"{'Bits':<6}"
      f"{'Levels':<8}"
      f"{'Delta':<12}"
      f"{'MSE':<15}"
      f"{'Measured SQNR':<18}"
      f"{'Theory SQNR':<15}"
      f"{'Difference':<12}")

print("-" * 85)

for row in results:
    print(f"{row[0]:<6}"
          f"{row[1]:<8}"
          f"{row[2]:<12.6f}"
          f"{row[3]:<15.8e}"
          f"{row[4]:<18.4f}"
          f"{row[5]:<15.4f}"
          f"{row[6]:<12.4f}")


# ------------------------------------------------------------
# 5. Display some PCM samples
# ------------------------------------------------------------

n_bits = 4

xq, indices, codewords, mse, sqnr, theoretical = \
    uniform_pcm_quantizer(x, n_bits)

print("\n" + "=" * 60)
print(f"FIRST 20 SAMPLES FOR {n_bits}-BIT PCM")
print("=" * 60)

print(f"{'Original':<15}"
      f"{'Quantized':<15}"
      f"{'Index':<10}"
      f"{'PCM Word':<10}")

print("-" * 60)

for i in range(20):
    print(f"{x[i]:<15.6f}"
          f"{xq[i]:<15.6f}"
          f"{indices[i]:<10}"
          f"{codewords[i]:<10}")


# ------------------------------------------------------------
# 6. Verify mandatory conditions
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

for n in range(2, 9):

    xq, indices, codewords, mse, sqnr, theoretical = \
        uniform_pcm_quantizer(x, n)

    L = 2 ** n

    # Check index range
    index_valid = (
        np.min(indices) >= 0 and
        np.max(indices) <= L - 1
    )

    # Check PCM word length
    word_length_valid = all(
        len(word) == n for word in codewords
    )

    print(
        f"{n}-bit: "
        f"Index range = {index_valid}, "
        f"PCM word length = {word_length_valid}"
    )


# ------------------------------------------------------------
# 7. Plot original and quantized waveform
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(t[:300], x[:300], label="Original Signal")
plt.step(t[:300], xq[:300], where='mid',
         label=f"{n_bits}-bit Quantized Signal")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Original vs {n_bits}-bit Quantized Signal")
plt.grid(True)
plt.legend()
plt.show()


# ------------------------------------------------------------
# 8. Plot quantization error
# ------------------------------------------------------------

error = x - xq

plt.figure(figsize=(10, 4))

plt.plot(t[:300], error[:300])

plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.title(f"Quantization Error ({n_bits}-bit PCM)")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 9. Quantization error histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(error, bins=30)

plt.xlabel("Quantization Error")
plt.ylabel("Number of Samples")
plt.title(f"Quantization Error Histogram ({n_bits}-bit)")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 10. SQNR versus number of bits
# ------------------------------------------------------------

bits = np.array([r[0] for r in results])
measured_sqnr = np.array([r[4] for r in results])
theoretical_sqnr = np.array([r[5] for r in results])

plt.figure(figsize=(9, 5))

plt.plot(bits, measured_sqnr, 'o-', label="Measured SQNR")
plt.plot(bits, theoretical_sqnr, 's--',
         label="Theoretical: 6.02n + 1.76 dB")

plt.xlabel("Number of Bits")
plt.ylabel("SQNR (dB)")
plt.title("SQNR vs Number of Quantization Bits")
plt.xticks(bits)
plt.grid(True)
plt.legend()
plt.show()