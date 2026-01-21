# About this file by Minglei You, Di He (Dee He) and Junzi Chen at 20250821:
# this file is the script to auto run on Pi, so that
# 1. it reads GPS signal from GMOUSE, 
# 2. and 4G/5G signals from Pluto, 
# 3. and record GPS and 4/5G signal to a csv file 
#############################
# Minimum Pi/Python Library installation to run this demo
#
# 1. installation of adi for pluto - check the separate instruction.
# (after the adi installed, do the following in the virtual environment, as Python in virt env. is a stand alone one, apt won't work, DO NOT INSTALL THEM VIA APT)
# 2. the following libs are required for csv logging (pandas), GPS signal parsing (pynmea2, pyserial), and LED blinking (lgpio):
#      $ pip install --upgrade pip pynmea2 pandas pyserial lgpio
# 
#############################
# Hardware connection map:
#
# 4G/5G SMA Antenna ---> Pluto <----(power cable USB A to micro USB) <--- Power Bank ---(type C cable)---> Pi 5 USB type C
#                              <--->(signal cable USB A to micro USB)<-----------------------------------> Pi 5 USB 3.0 (bottom blue usb socket, ttyACM1)
# GMOUSE GPS DONGLE --->(USB A cable)--------------------------------------------------------------------> Pi 5 USB 3.0 (top blue usb socket, ttyACM0) 
# LED DIODE Positive (longer leg) ------------>(female to female Dubang Jumper cable) -------------------> Pi 5 GPIO 17 (physical pin number 11)(1st physical pin is next to the J8 indicator, and the arrangement is the first row 1,2, second row 3,4, third row 5,6...)
#           Negative (shorter leg)------------>(female to female Dubang Jumper cable) -------------------> Pi 5 GND (physical pin number 9 or 14 or 20 or 25 or 34 or 39)(1st physical pin is next to the J8 indicator)
#
#############################
# Software diagram:
# 1. File structure:
#       to be added by DEE
# 2. Code structure:
#       to be added by DEE
# 3. Data structure:
#    (in the csv file)
#    ts, lat, lon, stat, {prn_1, snr_1},...,{prn_M, snr_M}, {rx_avg_power_1(linear),rx_avg_power_1(dB)},...,{rx_avg_power_N(linear),rx_avg_power_N(dB)}
#    
#    Legend:
#       ts:     time from GPS,  e.g., '20:25:27+00:00'
#       lat:    latitude,       e.g., 52.9779175
#       lon:    longitude,      e.g., -1.2236078333333
#       stat:   valid recording? if from RMC message, V means invalid(void) and A means valid; elif from GGA message, it is the number of satellite in vision, 0 means invalid, >=1 means valid.
#       {prn_n, snr_n}: for 1<=n<=N, there are N pairs of satellite indicated by their psedocode prn_n and their corresponding measured signal to noise ratio snr.
#       {rx_avg_power_m(linear),rx_avg_power_m(dB)}: for 1<=m<=M, there are M cellular bands observed (e.g., 4G and/or 5G).

# 4. Pi Auto run service configuration (running this script from powering it on):
#       the service for the auto-run file dir is as follows:
#       /etc/systemd/system/test2.service
# 
#
#
# Note from Ming 20250823-24:
# an update of the connection logic with glob lib, so that serial can pick up the correct device type.
# it is also noticed that when using AD9361 and 2t2r mode, the maximum sampling rate will be capped at 30MHz - so check the firmware if "error errno22 invalid parameter" 
#
# Note from Ming 20250821:
# this 20250821 version also adds led blinking for the ease of checking the status without screen.


# Note from Ming 20250821: 
# the gpsd method is abandoned due to the evaluation results that GPS freezing issue:
# the gpsd does not work, they give a constant GPS reading and no changing.
# this is cross-checked with Matlab and serial interface, where the raw NMEA data
# works perfectly with jittering observed, which indicates gpsd is with signal freezing issues.
# also it is noticed that in some tests, the gpsd will repeat the old readings (proof via the time stamp it reports).
# this might be some issue that can be solved later - but in this version, all gpsd related features are abandoned. 
# note that gps3 has been tested with the same issue, because they also call gpsd for the service.

# Note from Ming and Dee on 20250820:
# the service for the auto-run file dir is as follows:
# /etc/systemd/system/test2.service


import adi              # for Pluto
import numpy as np      # for fft
import pandas as pd     # for data saving
import serial           # for talking to gps
import pynmea2          # for parsing gps signal
import time             # for adding delays
import lgpio            # for led blinking as indicators
from datetime import datetime, date, timezone # for csv recording of gps time formatting to an easy format to deal with
import glob             # for serial connection by name 
#import matplotlib.pyplot as plt # for debugging


######################################################################
# this part is for the feature enablers

# debugging info output, mainly print
DBG_ENABLE = 0

# led blinking feature
LED_BLINKING_ENABLE = 1
if LED_BLINKING_ENABLE:
    CHIP = 0        # gpiochip0 for user GPIOs on Pi 5
    PIN = 17        # BCM 17 (physical pin 11)
    h = lgpio.gpiochip_open(CHIP)
    lgpio.gpio_claim_output(h, PIN)   # set as output
    # to blink the led connecting to the pin 11 with the negative leg on the gnd:
    # lgpio.gpio_write(h, PIN, 1)       # set HIGH
    # time.sleep(5)                     # keep it on for 5 seconds
    # lgpio.gpio_write(h, PIN, 0)       # set LOW
    # lgpio.gpiochip_close(h)

# gmouse setup for gps, config to 1 to activate
GPS_ENABLE = 1
# $ ls /dev/tty*
# Look for /dev/ttyACM1
# $ sudo cat /dev/ttyACM1
# Should see gmouse outputs, ctrl+c to exit
if GPS_ENABLE:
    try:
        time.sleep(5)
        gps_list = glob.glob('/dev/gps*')
        if not devices:
            raise RuntimeError("No GPS device found")
        gps_port = gps_list[0]
        ser = serial.Serial(gps_port, baudrate=9600, timeout=1)
    #    print("using /dev/ttyACM0")
    #    ser=serial.Serial('/dev/ttyACM0',9600, timeout=1)
    except Exception:
        time.sleep(5)
        # if it is not that port, it is this port, right? 
        print("using /dev/ttyACM1")
        ser=serial.Serial('/dev/ttyACM1',9600, timeout=1)
    # buffer clear to avoid freezing
    ser.reset_input_buffer()
    ser.reset_output_buffer()
######################################################################
# paramter initilization

# Mission parameters
fileName = "gps_debugging_20250821_v3.csv" # data logging file name, for the use with pd
append_frequency = 10 # record every 10 times - in case power failure or any interruption.
# total_data_count = 0 
# total_no_data =400  # 4000~4h this need to be calibrated @DeeHe if for a fixed amount of data

######################################################################
# SDR parameters
number_of_samples = 4096  #FFT points, smaller will be more transicent and observing partial packets, but fast; larger might take more time and memory space (even break);empirically set, this need to be calibrated if needed @DeeHe.
rx_gain = 50 # Pluto gain range 0-90dB, check the separate power calibration report generated by Ming for details.
sample_rate = int(40e6) # this is the baseband sample rate, in the unit of sample per second, each sample is an ADC observation in the baseband, formats in float32 or complex64 (e.g.,)
# Test set
# centrefrequency=[2.462e9];
# bandwidth=[20e6];
# 5G n78, order: Voldafone, Three, O2, EE
centrefrequency = [3.74e9, 3.47e9, 3.52e9, 3.56e9] # might need to check if there are uplink/downlink diff.
bandwidth = [20e6, 20e6, 20e6, 20e6] # the bandwidth might need to be double-checked
multi_sample_number = 10  # to take the pluto measurement 10 times and average out to allow a smooth result 
######################################################################
# House keeping variables
data_log = []  # data_set for the whole process
data_5G = []  # data_set for every loop
t = []  # time from gps reading
lat=[]  # latitude from gps reading
lon=[]  # longitude from gps reading
stat=[] # stat from gps reading - if from RMC message, V means invalid(void) and A means valid; elif from GGA message, it is the number of satellite in vision, 0 means invalid, >=1 means valid.


rx_avg_power = [np.nan]*2*len(centrefrequency)
######################################################################
print('let the game begin')

# Receive loop

try:
    while True:
    # while total_data_count<total_no_data: if for a fixed amount of data
    #########################################################################
        if GPS_ENABLE:
            
            # bug free for those recording lines without snr, 
            prn = [] # here prn mean psedocode index, which is used to refer back which protocol/satellite it is talking with.
            snr = [] # signal to noise ratio about the satellite link
            
            s = ser.readline().decode(errors='ignore').strip() # reading from the gps module via Serial interface, and decode it. Does it wait until reading a valid line? Does it block the code running for the Pluto (Pluto and GPS should be Asynchronised to ensure the max recording speed)? 
            
            #if DBG_ENABLE:
            #    print(s) 
            if not s.startswith('$'):
                # all response from the gmouse is starting with the indicator of $
                continue
            try:
                # pynmea2 to help parse and save craps
                m = pynmea2.parse(s)
            except Exception:
                continue

            if m.sentence_type == 'GGA':
                # Fix data. Gives UTC time, latitude, longitude, fix quality (0–8), satellites used, HDOP, altitude above mean sea level, and geoid separation. Core fix + altitude.
                # Basic fields
                t   = getattr(m, 'timestamp', None)         # time only (no date)
                lat = m.latitude
                lon = m.longitude
                #alt = getattr(m, 'altitude', None)          # meters above MSL
                stat = getattr(m, 'gps_qual', None)         # 0 = invalid, >=1, valid
                #sats = getattr(m, 'num_sats', None)
                #hdop = getattr(m, 'horizontal_dil', None)
                #geoid = getattr(m, 'geo_sep', None)
                if DBG_ENABLE:
                    print(f"GGA t={t} lat={lat} lon={lon} stat={stat}")
            elif m.sentence_type == 'RMC':
                t   = getattr(m, 'timestamp', None)
                lat  = m.latitude
                lon  = m.longitude
                #spdk = getattr(m, 'spd_over_grnd', None)  # knots
                #spdm = float(spdk)*0.514444 if spdk not in (None, '') else None
                #trk  = getattr(m, 'true_course', None)
                stat = getattr(m, 'status', None)           # A=valid, V= void   
                if DBG_ENABLE:
                    print(f"GGA t={t} lat={lat} lon={lon} stat={stat}")
                
            elif m.sentence_type == 'GSV':
                # Satellites in view. Lists all visible satellites with PRN, elevation, azimuth, and SNR (dB-Hz); 
                # often spans multiple sentences per epoch. Shows sky view & signal strengths (not which ones are used—that’s in GSA).
                # update SNR map for each satellite reported in this GSV sentence
                for i in range(1, 5):   # up to 4 sats per GSV line
                    prn = getattr(m, f"sv_prn_num_{i}", None)
                    snr = getattr(m, f"snr_{i}", None)
                    #if prn and snr not in (None, ''):
                    #    print("PRN", prn, "SNR(dB-Hz)", snr)
            
        #########################################################################
        # cellular signal scanning via Pluto
        for i in range(len(centrefrequency)): # to iterate over all assigned central frequencies
            # Setup SDR
            sdr = adi.Pluto("ip:192.168.2.1") # pluto to connect to ACM1, suggest to use dedicated power cable if Pi is on battery.
            sdr.rx_lo = int(centrefrequency[i])     # Hz, central frequency set as rx frequency
            sdr.rx_rf_bandwidth = int(bandwidth[i]) # Hz, bandwidth allocation - note that there could be filter issues if the bandwidth == sampling frequency; general rule is bandwidth <= sampling rate due to IIR;
            sdr.gain_control_mode_chan0 = "manual"  # manual needed to ensure no AGC applied
            sdr.rx_hardwaregain_chan0 = rx_gain     # dB, the gain is controlled - check the separate power calibration report generated by Ming for details.
            sdr.rx_buffer_size = number_of_samples  # the amount of data to be buffered/received every round of scanning
            sdr.sample_rate = sample_rate           # baseband sampling rate, sample/second
            #if DBG_ENABLE:
            #    print(sdr);
            valid_power = []
            
            # the following part might need more comments from Junzi 
            for j in range(multi_sample_number):
                rx_results = sdr.rx()
                if len(rx_results) != 0:
                    #if DBG_ENABLE:
                    #    print(rx_results.dtype)
                    #    plt.plot(range(len(rx_results)),abs(rx_results))
                    #    plt.show()
                    # Remove DC component, for the following FFT treatment
                    DC = np.mean(rx_results)
                    rx_results_demeaned = rx_results-DC
                    # FFT
                    rx_results_fft = np.fft.fft(rx_results_demeaned) # the fft stuff.
                    f_fft = np.fft.fftfreq(len(rx_results_demeaned), 1/sample_rate) # the corresponding frequency against the np.fft.fft result
                    # Shift FFT - by default fft is two-sided, this is to shift it to the real range, and certainly their corresponding frequency label f_fft need to align with the shift change
                    rx_results_fft_shifted = np.fft.fftshift(rx_results_fft)
                    f_fft_shifted = np.fft.fftshift(f_fft)
                    rx_results_fft_abs = abs(rx_results_fft_shifted) # complex to real?
                    #if DBG_ENABLE:
                    #    plt.plot(f_fft_shifted,rx_results_fft_abs)
                    #    plt.show()
                    # Calculate PSD [V**2/Hz], again, check the separate power calibration report generated by Ming for details? Might also need comments from Junzi 
                    rx_results_PSD = [x**2 for x in rx_results_fft_abs]
                    valid_power.append(np.mean(rx_results_PSD)/number_of_samples)
                    
            if len(valid_power):
                rx_avg_power[2*i] = max(valid_power) # linear format
                rx_avg_power[2*i+1] = 5 * np.log10(max(valid_power)/min(valid_power)) # dB format
            else:
                rx_avg_power[2*i] = np.nan # bug free
                rx_avg_power[2*i+1] = np.nan # bug free
            #if DBG_ENABLE:
            #   print('P_linear:', rx_avg_power[2*i])
            #   print('P_dB:', rx_avg_power[2*i+1])
            del sdr # is this necessary to delete it? is it possible to only shift the frequency - this might save some time????
            
        #########################################################################
        if GPS_ENABLE:
            ts = (t.isoformat() if hasattr(t, 'isoformat') else str(t)) # to remove the format for the csv logging, or it will be datetime.time(15, 28, 6, tzinfo=datetime.timezone.utc)
            new_row = [ts, lat, lon, stat] # time, latitude, longitude, status(whether this recording is valid, 0 or V for void/invalid, >=1 or A for valid gps recording)
            new_row.append(prn) # gps signal pseudo code index, to tell which satellite is in use
            new_row.append(snr) # and their corresponding signal to noise ratio (snr)
            for i in range(2*len(centrefrequency)):
                new_row.append(rx_avg_power[i].item()) # .item() to remove the np.float32 info for csv logging
            data_5G.append(new_row)
        else:
            data_5G.append(rx_avg_power.item()) # is this correct???
        #########################################################################
        # total_data_count+=1;
        if len(data_5G) >= append_frequency:  # if the recorded signals are more than this append_frequency number, then write it to the file!
            if LED_BLINKING_ENABLE:
                lgpio.gpio_write(h, PIN, 1) # it is a bit slow if blink is set here.
            # csv writing via pd
            data_log.append(data_5G)    
            data_pd = pd.DataFrame(data_log)
            data_pd.to_csv(fileName, mode='a', header=False, index=False)
            # bug free
            data_5G = []
        else:
            if LED_BLINKING_ENABLE:
                lgpio.gpio_write(h, PIN, 0) # it is a bit slow if blink is set here.

except KeyboardInterrupt:
    print("interupt..") # for debugging convenience
except Exception as e:
    print("Error:", e) # for any errors, throw the error message out

# Save remain data
finally: # work with try - it always runs after try code finishes
    if data_log: # if anything yet to write
        pd.DataFrame(data_log).to_csv(fileName, mode='a', header=False, index=False)

    # house cleaning - close the port just in case
    if LED_BLINKING_ENABLE:
        lgpio.gpiochip_close(h) 
    if GPS_ENABLE:
        ser.close()
    
    print("Game End - Good luck with the data")