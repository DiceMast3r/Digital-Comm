import numpy as np #ใช้สำหรับการคำนวณทางคณิตศาสตร์และการจัดการอาเรย์
import matplotlib.pyplot as plt #ใช้สำหรับการสร้างกราฟและแสดงข้อมูล
#from scipy.fft import fft, fftshift #ใช้สำหรับการคำนวณการเปลี่ยนฟูริเยร์ (FFT)
import scipy.fft as fft
#import cv2 #ไลบรารี OpenCV ใช้สำหรับการประมวลผลภาพ


# Constants
Eb = 1  # พลังงานต่อบิต
SNR_dBs = [3, 7, 10]  # SNR values in dB
distances = [10, 50, 100]  # Distances in meters
data_rates = [1e6, 2e6, 5e6]  # Data rates in bits per second (1Mbps, 2Mbps, 5Mbps)
path_loss_exponent = 3  # ตัวแปรที่ใช้คำนวณการสูญเสียสัญญาณตามระยะทาง

"""#invoke image from folder
img = cv2.imread("codeminidigi/orange.jpg")
img = cv2.resize(img,(200,200)) #เปลี่ยนขนาดภาพเป็น 200x200 พิกเซล
cv2.imshow("Original Picture",img) #แสดงภาพต้นฉบับ
#cv2.waitKey(0) #รอการกดปุ่มเพื่อปิดหน้าต่างภาพ

#Transform image to binary
img_RGB = np.array(img) #แปลงภาพเป็นอาเรย์ NumPy
img_binary =  np.unpackbits(img_RGB) #แปลงค่าพิกเซลเป็นบิต"""

"""bits = [] #สร้างลิสต์เพื่อเก็บบิตที่แปลงแล้ว
for i in img_binary:
    bits.append(i)
"""
np.random.seed(41) 
bits = np.random.randint(0, 2, 1000) 

N_bits = len(bits)
# Generate random bits


# Modulation schemes
def qpsk_modulation(bits): #สร้างฟังก์ชันชื่อ qpsk_modulation ที่รับ input เป็นลิสต์ของ bits
    symbols = [] #สร้างลิสต์ว่างชื่อ symbols เพื่อใช้เก็บสัญลักษณ์ที่ได้จากการแมปคู่บิต 
    for i in range(0, len(bits), 2): #วนอ่านบิตในลิสต์ bits ทีละ 2 ตำแหน่ง ลูปนี้จะเริ่มจากตำแหน่งที่ 0 และเพิ่มขึ้นทีละ 2 จนถึงบิตสุดท้าย
        if bits[i] == 0 and bits[i+1] == 0: #ถ้าบิต 2 บิตที่อ่านได้มีค่าเป็น 0, 0
            symbols.append(-1 - 1j) #เพิ่มสัญลักษณ์ 1-1j ลงในลิสต์ symbols (Quadrant III (มุม 225°) ของคอนสเตลเลชัน QPSK)
        elif bits[i] == 0 and bits[i+1] == 1: #ถ้าบิตมีค่าเป็น 0, 1
            symbols.append(-1 + 1j) #เพิ่มสัญลักษณ์ -1+1j ลงในลิสต์ symbols (Quadrant II (มุม 135°))
        elif bits[i] == 1 and bits[i+1] == 0: #ถ้าบิตมีค่าเป็น 1, 0
            symbols.append(1 - 1j) #เพิ่มสัญลักษณ์ 1-1j ลงในลิสต์ symbols (Quadrant IV (มุม 315°))
        else:                       #ถ้าบิตมีค่าเป็น 1, 1
            symbols.append(1 + 1j) #เพิ่มสัญลักษณ์ 1+1j ลงในลิสต์ symbols (Quadrant I (มุม 45°))
    return np.array(symbols) #แปลงลิสต์ของสัญลักษณ์เป็น NumPy array และคืนค่าอาเรย์ของสัญลักษณ์ QPSK เป็นผลลัพธ์ของฟังก์ชัน


def psk_modulation(bits, M): #สร้างฟังก์ชันชื่อ psk_modulation ซึ่งรับพารามิเตอร์ 2 ตัว คือ bits (ลิสต์ของบิตที่ต้องการมอดูเลต) 
                                                                            #M (จำนวนระดับเฟส 4 สำหรับ QPSK)
    k = int(np.log2(M)) #ใช้ np.log2(M) เพื่อตรวจสอบว่าต้องใช้บิตกี่บิตต่อสัญลักษณ์
    symbols = [] #สร้างลิสต์ว่างชื่อ symbols เพื่อเก็บสัญลักษณ์ที่ได้จากการมอดูเลต
    for i in range(0, len(bits), k): #ใช้ for loop เพื่อวนอ่านบิตทีละ k บิต
        index = int(''.join(map(str, bits[i:i+k])), 2) #แปลงบิตเป็นเลขฐานสิบ
        symbol = np.exp(1j * 2 * np.pi * index / M) #คำนวณสัญลักษณ์ PSK โดยใช้เลขฐานสิบ
        symbols.append(symbol) #ช้ np.array() เพื่อแปลงลิสต์ symbols ให้เป็น NumPy array ก่อนคืนค่า
    return np.array(symbols)


# BER Calculation
def calculate_ber(original_bits, received_bits): #สร้างฟังก์ชันชื่อ calculate_ber ซึ่งรับพารามิเตอร์ 2 ตัว คือ original_bits: ลิสต์ของบิตต้นฉบับ (บิตที่ส่งออกไป)
                                                                                                #received_bits: ลิสต์ของบิตที่รับกลับมา (ซึ่งอาจมีข้อผิดพลาด)

    return np.sum(original_bits != received_bits) / len(original_bits) #นับจำนวนบิตต้นทางไม่ตรงกับบิตที่รับกลับมาหารกับความยาวของบิตต้นทาง


# Path loss calculation based on distance
def calculate_path_loss(d, path_loss_exponent): #สร้างฟังก์ชันชื่อ calculate_path_loss ซึ่งรับพารามิเตอร์ 2 ตัว คือ d: ระยะทางระหว่างเครื่องส่งกับเครื่องรับ (มีหน่วยเป็นเมตร)
                                                                                            #path_loss_exponent: ตัวเลขที่แสดงระดับการลดทอนของสัญญาณ(เช่น 2 ในกรณี free-space)
    return 10 * path_loss_exponent * np.log10(d) #สมการคำนวณ pass loss หน่วยdB อิงจากสูตร pass loss=10*n*log10(d)
                                                    #n คือ path loss exponent

# AWGN Noise
def awgn(signal, snr_db): #สร้างฟังก์ชันชื่อ awgn ซึ่งรับพารามิเตอร์ 2 ตัว คือ signal (อาเรย์ของสัญญาณที่ต้องการเพิ่มสัญญาณรบกวน) , snr_db (อัตราส่วนสัญญาณต่อสัญญาณรบกวน)
    snr = 10**(snr_db/10) #แปลง SNR จากหน่วย dB เป็นอัตราส่วนเชิงเส้น
    noise_std = np.sqrt(1/(2 * snr)) #คำนวณค่าเบี่ยงเบนมาตรฐานของสัญญาณรบกวน
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) #สร้างสัญญาณรบกวนแบบ AWGN
    return signal + noise #รวมสัญญาณรบกวนเข้ากับสัญญาณต้นฉบับและคืนค่าสัญญาณที่มี AWGN

                                                                                #original_bits: บิตต้นฉบับที่ถูกส่งออกไป (ใช้เปรียบเทียบเพื่อคำนวณ BER)
# Demodulation and BER Calculation                                              #signal: สัญญาณที่รับกลับมาหลังจากผ่านช่องทางการสื่อสาร (มีสัญญาณรบกวน)
def demodulate_and_calculate_ber(modulation_scheme, signal, original_bits, M):  #modulation_scheme (ชื่อของเทคนิคการมอดูเลตที่ใช้ เช่น 'qpsk')  #M: จำนวนสัญลักษณ์ใน M-ary PSK
    if modulation_scheme == 'qpsk': #เช็คว่ารูปแบบการมอดูเลตคือ QPSK หรือไม่ ถ้าใช่จะสร้างลิสต์เพื่อเก็บผลลัพธ์การถอดรหัสบิต                                           
        demodulated_bits = []
        for s in signal: #วนลูป ผ่านแต่ละสัญลักษณ์เชิงซ้อนใน signal
            real_part, imag_part = np.real(s), np.imag(s) #แยกส่วนจริง (real_part) และส่วนจินตภาพ (imag_part)
            if real_part < 0: #ใช้เงื่อนไข if-else เพื่อตรวจสอบค่าส่วนจริงและจินตภาพ
                if imag_part < 0:
                    demodulated_bits += [0, 0] #(Real<0,Imag<0) → บิต [0, 0]
                else:
                    demodulated_bits += [0, 1] #(Real<0,Imag≥0) → บิต [0, 1]
            else:
                if imag_part < 0:
                    demodulated_bits += [1, 0] #(Real≥0,Imag<0) → บิต [1, 0]
                else:
                    demodulated_bits += [1, 1] #(Real≥0,Imag≥0) → บิต [1, 1]
    else:                           #ถ้าไม่ใช่ QPSK จะใช้วิธี M-ary PSK โดยเริ่มต้นด้วยการสร้างลิสต์ demodulated_bits
        demodulated_bits = []
        for s in signal:
            phase_angle = np.angle(s) #np.angle(s): คำนวณเฟสของสัญลักษณ์เชิงซ้อน s
            index = int(np.round(M * phase_angle / (2 * np.pi)) % M) #คำนวณ index ของสัญลักษณ์จากเฟสที่ได้
            demodulated_bits += list(map(int, bin(index)[2:].zfill(int(np.log2(M))))) #แปลง index เป็นบิต
    demodulated_bits = np.array(demodulated_bits[:len(original_bits)]) #ตัดบิตส่วนเกินออกให้มีความยาวเท่ากับ original_bitsและแปลงเป็น numpy array
    ber = np.sum(demodulated_bits != original_bits) / len(original_bits) #เปรียบเทียบบิตที่ถอดรหัสได้กับ original_bits
    return ber #คืนค่า BER ที่คำนวณได้


# Plot signal in time domain
def plot_time_domain(signal, title): #signal: อาเรย์ของสัญญาณเชิงซ้อนที่ต้องการแสดงผล , title: ชื่อกราฟที่จะแสดงเป็นหัวข้อ
    plt.figure()
    plt.plot(np.real(signal[:100]), label="Real part") #ดึงส่วนจริงของสัญญาณมาพล็อตแค่100จุดแรก
    plt.plot(np.imag(signal[:100]), label="Imaginary part") ##ดึงส่วนจินตภาพของสัญญาณมาพล็อตแค่100จุดแรก
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    
    
# Plot signal in frequency domain
def plot_frequency_domain(signal, title): #signal: สัญญาณใน Time Domain ที่ต้องการแปลงไปเป็น Frequency Domain
    signal_fft = np.array(fft.fft(signal)) #คำนวณ Fast Fourier Transform (FFT) ของสัญญาณ (แปลงจากtimeเป็นfreq)
    freqs = np.fft.fftfreq(len(signal), d=1) # คำนวณความถี่ที่สอดคล้องกับผลลัพธ์ FFT (len(signal)=จำนวนจุดข้อมูล) (กำหนดระยะห่างตัวอย่างข้อมูลคือ1หน่วยเวลา)
    plt.figure()
    plt.plot(freqs, np.abs(signal_fft)) #คำนวณขนาด (Magnitude) ของแต่ละค่าความถี่
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid()
    
    
# Main simulation loop
ber_qpsk_list = []
ber_8psk_list = []
ber_16psk_list = []
eb_no_values = []


for snr_db in SNR_dBs:
    # Calculate Eb/No from SNR for each modulation
    M_qpsk = 4
    M_8psk = 8
    M_16psk = 16
    eb_no_qpsk = snr_db - 10 * np.log10(np.log2(M_qpsk)) 
    eb_no_8psk = snr_db - 10 * np.log10(np.log2(M_8psk))
    eb_no_16psk = snr_db - 10 * np.log10(np.log2(M_16psk))
    eb_no_values.append((eb_no_qpsk, eb_no_8psk, eb_no_16psk))
    
    # QPSK
    qpsk_symbols = qpsk_modulation(bits)
    qpsk_noisy = awgn(qpsk_symbols, snr_db) #เพิ่ม noise
    qpsk_signal = qpsk_symbols
    fig, xis = plt.subplots(2,2,figsize=(12,6)) #กราฟกว้าง12หน่วยสูง6หน่วย
    signal1 = np.real(qpsk_symbols[:100]) #ส่วนจริง100จุดแรก
    signal2 = np.imag(qpsk_symbols[:100]) #ส่วนจินตภาพ100จุดแรก
    xis[0,0].plot(signal1) #พล็อตส่วนจริงของQPSK
    xis[0,0].grid(True)
    xis[0,0].set_title(f"QPSK (real) Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,0].set_xlabel('Time')
    xis[0,0].set_ylabel("Amplitude")
    
    xis[0,1].plot(signal2) #พล็อตส่วนจินตภาพของQPSK
    xis[0,1].grid(True)
    xis[0,1].set_title(f"QPSK (img) Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,1].set_xlabel('Time')
    xis[0,1].set_ylabel("Amplitude")
    
    xis[1,0].plot(qpsk_signal) #พล็อตQPSKที่มีสัญญาณรบกวน
    xis[1,0].grid(True)
    xis[1,0].set_title(f"QPSK Time Domain (No Noise, SNR={snr_db} dB)")
    xis[1,0].set_xlabel('Time')
    xis[1,0].set_ylabel("Amplitude")
    
    qpsk_noisy_fft = np.array(fft.fft(qpsk_noisy)) # คำนวณ FFT ของสัญญาณที่มี Noise
    freqs= np.fft.fftfreq(len(qpsk_noisy),d=1) #คำนวณความถี่ที่สอดคล้องกับผลลัพธ์ FFT
    xis[1,1].plot(freqs , qpsk_noisy_fft) #Plot สัญญาณใน Frequency Domain
    xis[1,1].grid(True)
    xis[1,1].set_title(f"QPSK Frequency Domain (SNR={snr_db} dB)")
    xis[1,1].set_xlabel('Frequency')
    xis[1,1].set_ylabel("Magnitude")
    
    
    ber_qpsk = demodulate_and_calculate_ber('qpsk', qpsk_noisy, bits, M_qpsk) #เรียกใช้ฟังก์ชันเพื่อทำการ Demodulate สัญญาณ QPSK ที่มีสัญญาณรบกวน
    ber_qpsk_list.append(ber_qpsk) #คำนวณ ber แล้วเก็บไว้

    # 8-PSK
    psk8_symbols = psk_modulation(bits, M_8psk)
    fig, xis = plt.subplots(2,2,figsize=(12,6))
    psk8_noisy = awgn(psk8_symbols, snr_db)
    signal1 = np.real(psk8_symbols[:100])
    signal2 = np.imag(psk8_symbols[:100])
    xis[0,0].plot(signal1)
    xis[0,0].grid(True)
    xis[0,0].set_title(f"8-PSK Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,0].set_xlabel('Time')
    xis[0,0].set_ylabel("Amplitude")
    
    xis[0,1].plot(signal2)
    xis[0,1].grid(True)
    xis[0,1].set_title(f"8-PSK (img) Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,1].set_xlabel('Time')
    xis[0,1].set_ylabel("Amplitude")
    
    xis[1,0].plot(psk8_noisy)
    xis[1,0].grid(True)
    xis[1,0].set_title(f"8-PSK Time Domain (With Noise, SNR={snr_db} dB)")
    xis[1,0].set_xlabel('Time')
    xis[1,0].set_ylabel("Amplitude")
    
    psk8_noisy_fft = np.array(fft.fft(psk8_noisy))
    freqs= np.fft.fftfreq(len(psk8_noisy),d=1)
    xis[1,1].plot(freqs , psk8_noisy_fft)
    xis[1,1].grid(True)
    xis[1,1].set_title(f"8-PSK Frequency Domain (SNR={snr_db} dB)")
    xis[1,1].set_xlabel('Frequency')
    xis[1,1].set_ylabel("Magnitude")

    ber_8psk = demodulate_and_calculate_ber('8psk', psk8_noisy, bits, M_8psk)
    ber_8psk_list.append(ber_8psk)
    
    # 16-PSK
    psk16_symbols = psk_modulation(bits, M_16psk)
    psk16_noisy = awgn(psk16_symbols, snr_db)
    fig, xis = plt.subplots(2,2,figsize=(12,8))
    signal1 = np.real(psk16_symbols[:100])
    signal2 = np.imag(psk16_symbols[:100])
    xis[0,0].plot(signal1)
    xis[0,0].grid(True)
    xis[0,0].set_title(f"16-PSK Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,0].set_xlabel('Time')
    xis[0,0].set_ylabel("Amplitude")
    
    xis[0,1].plot(signal2)
    xis[0,1].grid(True)
    xis[0,1].set_title(f"16-PSK (img) Time Domain (No Noise, SNR={snr_db} dB)")
    xis[0,1].set_xlabel('Time')
    xis[0,1].set_ylabel("Amplitude")
    
    xis[1,0].plot(psk16_noisy)
    xis[1,0].grid(True)
    xis[1,0].set_title(f"16-PSK Time Domain (With Noise, SNR={snr_db} dB)")
    xis[1,0].set_xlabel('Time')
    xis[1,0].set_ylabel("Amplitude")
    
    psk16_noisy_fft = np.array(fft.fft(psk16_noisy))
    freqs= np.fft.fftfreq(len(psk16_noisy),d=1)
    xis[1,1].plot(freqs , psk16_noisy_fft)
    xis[1,1].grid(True)
    xis[1,1].set_title(f"16-PSK Frequency Domain (SNR={snr_db} dB)")
    xis[1,1].set_xlabel('Frequency')
    xis[1,1].set_ylabel("Magnitude")

    ber_16psk = demodulate_and_calculate_ber('16psk', psk16_noisy, bits, M_16psk)
    ber_16psk_list.append(ber_16psk)

    # Plot separate constellation diagrams
    fig , axis = plt.subplots(3,1,figsize=(8,10)) #สร้างกราฟ subplot ขนาด 3x1 (3 แถว1 คอลัมน์) ขนาดของกราฟให้กว้าง 8 หน่วย และสูง 10 หน่วย
    sig1 = np.real(qpsk_noisy) #คำนวณส่วนจริงของสัญญาณ QPSK ที่มีnoise แล้วเก็บไว้
    sig2 = np.imag(qpsk_noisy) #คำนวณส่วนจินตภาพของสัญญาณ QPSK ที่มีnoise แล้วเก็บไว้
    axis[0].scatter(sig1,sig2) #สร้าง scatter plot ของ QPSK Constellation
    axis[0].grid(True)
    axis[0].set_title(f"QPSK Constellation at SNR={snr_db} dB")
    

    sig1 = np.real(psk8_noisy)
    sig2 = np.imag(psk8_noisy)
    axis[1].scatter(sig1,sig2) #สร้าง scatter plot ในกราฟย่อยที่ตำแหน่ง (1)
    axis[1].grid(True)
    axis[1].set_title(f"8-PSK Constellation at SNR={snr_db} dB")

    sig1 = np.real(psk16_noisy)
    sig2 = np.imag(psk16_noisy)
    axis[2].scatter(sig1,sig2)
    axis[2].grid(True)
    axis[2].set_title(f"16-PSK Constellation at SNR={snr_db} dB")

    plt.show()
    ber_results = {} #สร้างตัวแปร ber_results เป็น Dictionary เปล่า ซึ่งจะใช้เก็บผลลัพธ์ Bit Error Rate (BER) ของแต่ละโมดูเลชันในภายหลัง
    
    
for data_rate in data_rates: #วนลูปผ่านค่าต่างๆของdata_rates
    ber_results[data_rate] = {} #ber_results[data_rate] จะเก็บค่าผลลัพธ์ BER สำหรับอัตราการส่งข้อมูลแต่ละค่า
    for distance in distances: #วนลูปผ่านค่าต่างๆของdistances
        ber_results[data_rate][distance] = [] #สร้างลิสต์เปล่าสำหรับเก็บค่า BER ในแต่ละระยะทางและอัตราการส่งข้อมูล
        for snr_db in SNR_dBs: #วนลูปผ่านค่า SNR_dBs
            # Path loss affects received power
            path_loss_dB = calculate_path_loss(distance, path_loss_exponent)
            effective_snr_db = snr_db - path_loss_dB #คำนวณ SNR ที่มีประสิทธิภาพ โดยการลบ path_loss_dB ออกจาก snr_db
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulation(bits) #โมดูเลตสัญญาณโดยใช้ QPSKเพื่อโมดูเลตบิตต้นฉบับ
            qpsk_noisy = awgn(qpsk_symbols, effective_snr_db) # เพิ่มสัญญาณรบกวน (AWGN) ลงในสัญญาณ QPSK
            # Demodulate and calculate BER
            demodulated_bits = np.array([0 if np.real(s) < 0 else 1 for s in qpsk_noisy]) #คำนวณค่าจากส่วนจริงของแต่ละสัญญาณใน qpsk_noisy
                                                                                        #โดยเงื่อนไขคือ ถ้าส่วนจริง < 0 ให้เป็น 0,ถ้าส่วนจริง >= 0 ให้เป็น 1
            ber = calculate_ber(bits[:len(demodulated_bits)], demodulated_bits)
            ber_results[data_rate][distance].append(ber) #เพิ่มค่า ber ลงในลิสต์ ทำให้สามารถเก็บผลลัพธ์ BER สำหรับทุกค่าของ SNRที่วนลูป
            
            
# Plot BER vs. Eb/No for each modulation
eb_no_values = np.array(eb_no_values) # แปลง eb_no_values เป็น NumPy Array
plt.figure() #พล็อตกราฟ BER สำหรับ QPSK ('o-' หมายถึงใช้รูปแบบจุดและเส้นเชื่อมกัน)
plt.semilogy(eb_no_values[:, 0], ber_qpsk_list, 'o-', label="QPSK")
plt.semilogy(eb_no_values[:, 1], ber_8psk_list, 'o-', label="8-PSK")
plt.semilogy(eb_no_values[:, 2], ber_16psk_list, 'o-', label="16-PSK")
plt.xlabel("Eb/No (dB)")
plt.ylabel("BER")
plt.title("BER vs. Eb/No")
plt.legend()
plt.grid()
plt.show()


# Plot BER vs Eb/No for each distance and data rate
for data_rate in data_rates: #วนลูปเพื่อทำงานกับทุกค่าใน data_rates
    plt.figure()
    for distance in distances:
        plt.semilogy(SNR_dBs, ber_results[data_rate][distance], label=f"Distance={distance}m")
            #ใช้ฟังก์ชัน plt.semilogy(...) เพื่อพล็อตกราฟที่มีแกน y เป็นสเกลลอการิธึม
            #ber_results[data_rate][distance]: ค่าของ BER สำหรับอัตราข้อมูลและระยะทางที่กำหนด
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR at Different Distances (Data Rate={data_rate/1e6} Mbps)") #data_rate/1e6 แปลงค่าอัตราข้อมูลจากบิตต่อวินาที (bps) เป็นเมกะบิตต่อวินาที (Mbps)
    plt.legend()
    plt.grid()
    plt.show()