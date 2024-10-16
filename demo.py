import numpy as np
import matplotlib.pyplot as plt

# รับค่า student ID สองค่า
#student_ID1 = int(input('Enter your first student ID: '))
#student_ID2 = int(input('Enter your second student ID: '))

student_ID1 = 65010426
student_ID2 = 65010434

# คำนวณ A1 และ A2 โดยเอาหลักสุดท้ายของแต่ละ student ID
A1 = student_ID1 % 10
A2 = student_ID2 % 10

# ผลบวกของหลักสุดท้ายจาก student ID ทั้งสองค่า (x)
x = float(str(A1) + str(A2)) / 100  # เช่น 1.23 ถ้า A1=2, A2=3

# ตั้งค่าเริ่มต้น
Nbits = 1000000  # จำนวนบิตที่มากขึ้น
Nsamp = 10  # จำนวนตัวอย่างต่อบิต
np.random.seed(30)
a = np.random.randint(0, 2, Nbits)  # สุ่มบิต
b = 2 * a - 1  # แปลงบิตจาก {0, 1} เป็น {-1, 1}
Eb = 10  # พลังงานต่อบิต

# สร้างค่า SNRdB_log ตั้งแต่ 1.23 ถึง 9.23
#SNRdB_log = np.array([y + x for y in range(1, 10)])  # ผลลัพธ์จะเป็น [1.23, 2.23, ..., 9.23]
SNRdB_log = np.arange(1.23, 10.23, 1)  # ผลลัพธ์จะเป็น [1.23, 2.23, ..., 9.23]
SNRdB_log = 10 ** (SNRdB_log / 10)  # แปลง dB เป็น linear scale

def calculate_ber(snr_dB):
    # คำนวณค่า sigma จาก SNR
    sigma = np.sqrt(Eb / (2 * (10**(snr_dB / 10))))
    
    # Generate Differential NRZ Modulated signals
    x_t = []
    current_level = 1  # เริ่มต้นที่ 1
    for i in range(Nbits):
        if a[i] == 1:
            current_level *= -1  # พลิกสัญญาณเมื่อเจอบิต 1
        x_t.extend([current_level] * Nsamp)

    # Generate AWGN
    mu = 0
    n_t = np.random.normal(mu, sigma, Nbits * Nsamp)

    # Received signals
    r_t = np.array(x_t) + n_t

    # Correlator
    s_DNRZ = np.array([1] * Nsamp)  # for D-NRZ
    z = []

    for i in range(Nbits):
        z_t = np.multiply(r_t[i * Nsamp:(i + 1) * Nsamp], s_DNRZ)
        z_t_out = sum(z_t)
        z.append(z_t_out)

    # Make decision, compare z with 0
    a_hat = [1 if zdata > 0 else 0 for zdata in z]

    # Calculate error
    err_num = sum(a != a_hat)

    # Calculate BER
    ber = err_num / Nbits
    return ber

# Calculate BER for each SNR value for D-NRZ
a_BER_DNRZ = [calculate_ber(snr) for snr in SNRdB_log]

# Print BER values
for snr, ber in zip(SNRdB_log, a_BER_DNRZ):
    print(f"SNR (dB): {snr}, BER: {ber}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(SNRdB_log, a_BER_DNRZ, marker='o', label='D-NRZ')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True, which='both')
plt.title('BER vs. SNR for D-NRZ')

plt.show()