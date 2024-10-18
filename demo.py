import numpy as np
import matplotlib.pyplot as plt

# การกำหนดพารามิเตอร์
nbit = 10000  # จำนวนบิตข้อมูล
nsamp = 8    # จำนวนตัวอย่างต่อสัญญาณหนึ่งตัวอย่าง (oversampling factor)
M = 4        # ระดับของ PAM
Es = 1       # พลังงานเฉลี่ยของสัญญาณ

# การสุ่มบิตข้อมูล
data_bits = np.random.randint(0, M, nbit)

# การแมปบิตไปเป็นสัญญาณ 4 ระดับ (ระดับของ PAM: -3, -1, 1, 3)
pam_symbols = 2 * data_bits - (M - 1)

# การสร้างสัญญาณ 4-PAM (Modulation)
pam_signal = np.repeat(pam_symbols, nsamp)

# การเพิ่ม AWGN เข้าไปในสัญญาณ
SNR_dB = 10  # อัตราส่วนสัญญาณต่อเสียงรบกวน (Signal-to-Noise Ratio) ใน dB
SNR_linear = 10**(SNR_dB/10)
noise_std = np.sqrt(Es / (2 * SNR_linear))  # คำนวณค่าเบี่ยงเบนมาตรฐานของ AWGN
awgn_noise = noise_std * np.random.normal(0, 10, len(pam_signal))

received_signal = pam_signal + awgn_noise  # สัญญาณที่ได้รับพร้อม AWGN

# การทำ Demodulation
# ทำการ Decimation โดยการเฉลี่ยตัวอย่างในแต่ละช่วง (แต่ละสัญญาณ)
received_symbols = received_signal.reshape(-1, nsamp).mean(axis=1)

# การตัดสินใจ (Decision) ว่าสัญญาณที่ได้รับเป็นค่าใดใน PAM (ระดับ -3, -1, 1, 3)
demod_data_bits = np.zeros_like(received_symbols)

demod_data_bits[received_symbols >= 2] = 3
demod_data_bits[np.logical_and(received_symbols >= 0, received_symbols < 2)] = 2
demod_data_bits[np.logical_and(received_symbols >= -2, received_symbols < 0)] = 1
demod_data_bits[received_symbols < -2] = 0

# การคำนวณ Bit Error Rate (BER)
bit_errors = np.sum(data_bits != demod_data_bits)
BER = bit_errors / nbit

# การแสดงผลลัพธ์
print(f'Bit Error Rate (BER): {BER:.2f}')

# พล็อต data bit และข้อมูลอื่นๆ

plt.figure(figsize=(12, 10))

# พล็อตบิตข้อมูลที่สุ่มได้
plt.subplot(4, 1, 1)
plt.plot(data_bits[:100], 'o-', label="Data Bits")
plt.title('Random Data Bits')
plt.xlabel('Bit Index')
plt.ylabel('Bit Value')
plt.grid(True)
plt.legend()

# พล็อตสัญญาณที่ถูกส่ง (Transmitted 4-PAM Signal)
plt.subplot(4, 1, 2)
plt.plot(pam_signal[:100], label='Transmitted 4-PAM Signal', drawstyle='steps-post')
plt.title('Transmitted 4-PAM Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# พล็อตสัญญาณที่ได้รับ (Received Signal with AWGN)
plt.subplot(4, 1, 3)
plt.plot(received_signal[:100], label='Received Signal (with AWGN)', alpha=0.7)
plt.title('Received Signal (with AWGN)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# พล็อตสัญญาณหลังจากการทำ demodulation
plt.subplot(4, 1, 4)
plt.plot(demod_data_bits[:100], 'o-', label='Demodulated Symbols', drawstyle='steps-post')
plt.title('Demodulated 4-PAM Symbols')
plt.xlabel('Symbol Index')
plt.ylabel('Symbol Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# การสร้างกราฟ BER เทียบกับ SNR
SNR_dB_values = np.arange(-10, 10, 1)  # SNR ใน dB ตั้งแต่ -10 ถึง 10
BER_values = []

for SNR_dB in SNR_dB_values:
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_std = np.sqrt((1 / 2) * (10) * (10 ** (-SNR_dB / 10)))  # คำนวณค่าเบี่ยงเบนมาตรฐานของ AWGN
    #awgn_noise = noise_std * np.random.normal(0, 1, len(pam_signal))
    awgn_noise = np.random.normal(0, noise_std, len(pam_signal))

    received_signal = pam_signal + awgn_noise  # สัญญาณที่ได้รับพร้อม AWGN

    # การทำ Demodulation
    received_symbols = received_signal.reshape(-1, nsamp).mean(axis=1)

    demod_data_bits = np.zeros_like(received_symbols)
    demod_data_bits[received_symbols >= 2] = 3
    demod_data_bits[np.logical_and(received_symbols >= 0, received_symbols < 2)] = 2
    demod_data_bits[np.logical_and(received_symbols >= -2, received_symbols < 0)] = 1
    demod_data_bits[received_symbols < -2] = 0

    # การคำนวณ Bit Error Rate (BER)
    bit_errors = np.sum(data_bits != demod_data_bits)
    BER = bit_errors / nbit
    BER_values.append(BER)

# พล็อตกราฟ BER เทียบกับ SNR
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_dB_values, BER_values, 'o-', label='BER vs SNR for 4-PAM')
#plt.yscale('log')  # ใช้ logarithmic scale บนแกน Y
plt.title('BER vs SNR for 4-PAM with AWGN')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True)
plt.legend()
plt.show()