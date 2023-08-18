from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
import threading
import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

'''
GUI made by Fermín Rodríguez.

Guitar tuner script based on the Harmonic Product Spectrum (HPS)
MIT License
Copyright (c) 2021 chciken
check his post on the matter here: https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html
'''

class Circulito(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            self.color = Color(0,0,0)

class Circulon(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            self.color = Color(0.80,0.49,0.19)
    def cambiarColor(self):
        with self.canvas:
            self.color = Color(1,1,1)

    
class TunerWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notas = self.ids._notasAfinador
        # General settings that can be changed by the user
        self.SAMPLE_FREQ = 48000 # sample frequency in Hz
        self.WINDOW_SIZE = 48000 # window size of the DFT in samples
        self.WINDOW_STEP = 12000 # step size of window
        self.NUM_HPS = 5 # max number of harmonic product spectrums
        self.POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
        self.CONCERT_PITCH = 440 # defining a1
        self.WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off
        self.WINDOW_T_LEN = self.WINDOW_SIZE / self.SAMPLE_FREQ # length of the window in seconds
        self.SAMPLE_T_LENGTH = 1 / self.SAMPLE_FREQ # length between two samples in seconds
        self.DELTA_FREQ = self.SAMPLE_FREQ / self.WINDOW_SIZE # frequency step width of the interpolated DFT
        self.OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
        self.ALL_NOTES = ["LA","LA#","Si","Do","Do#","Re","Re#","Mi","Fa","Fa#","Sol","Sol#"]
        self.HANN_WINDOW = np.hanning(self.WINDOW_SIZE)
        self.noteBuffer = ["1", "2"]
        self.window_samples = [0 for _ in range(self.WINDOW_SIZE)]

    def find_closest_note(self,pitch):
        i = int(np.round(np.log2(pitch/self.CONCERT_PITCH)*12))
        closest_note = self.ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
        closest_pitch = self.CONCERT_PITCH*2**(i/12)
        return closest_note, closest_pitch

    def callback(self, indata, frames, time, status):
        if status:
            self.notas.text=status
            return
        if any(indata):
            self.window_samples = np.concatenate((self.window_samples, indata[:, 0]))
            self.window_samples = self.window_samples[len(indata[:, 0]):]

            # skip if signal power is too low
            signal_power = (np.linalg.norm(self.window_samples, ord=2)**2) / len(self.window_samples)
            if signal_power < self.POWER_THRESH:
                os.system('cls' if os.name=='nt' else 'clear')
                self.notas.text="¡Tocá che!"
                return

            # avoid spectral leakage by multiplying the signal with a hann window
            hann_samples = self.window_samples * self.HANN_WINDOW
            magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

            # supress mains hum, set everything below 62Hz to zero
            for i in range(int(62/self.DELTA_FREQ)):
                magnitude_spec[i] = 0

            # calculate average energy per frequency for the octave bands
            # and suppress everything below it
            for j in range(len(self.OCTAVE_BANDS)-1):
                ind_start = int(self.OCTAVE_BANDS[j]/self.DELTA_FREQ)
                ind_end = int(self.OCTAVE_BANDS[j+1]/self.DELTA_FREQ)
                ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
                avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end-ind_start)
                avg_energy_per_freq = avg_energy_per_freq**0.5
                for i in range(ind_start, ind_end):
                    magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > self.WHITE_NOISE_THRESH*avg_energy_per_freq else 0

            # interpolate spectrum
            mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/self.NUM_HPS), np.arange(0, len(magnitude_spec)),
                                    magnitude_spec)
            mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2) #normalize it

            hps_spec = copy.deepcopy(mag_spec_ipol)

            # calculate the HPS
            for i in range(self.NUM_HPS):
                tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
                if not any(tmp_hps_spec):
                    break
                hps_spec = tmp_hps_spec

            max_ind = np.argmax(hps_spec)
            max_freq = max_ind * (self.SAMPLE_FREQ/self.WINDOW_SIZE) / self.NUM_HPS

            closest_note, closest_pitch = self.find_closest_note(max_freq)
            max_freq = round(max_freq, 1)
            closest_pitch = round(closest_pitch, 1)

            self.noteBuffer.insert(0, closest_note) # note that this is a ringbuffer
            self.noteBuffer.pop()

            if self.noteBuffer.count(self.noteBuffer[0]) == len(self.noteBuffer):
                self.notas.text=f"{closest_note} {max_freq}/{closest_pitch}"
            else:
                self.notas.text=f""
        else:
            self.notas.text='¡Tocá che!'

    def start_stream(self):
        try:
            self.notas.text = "Iniciando el afinador tanguero..."
            audio_thread = threading.Thread(target=self.audio_stream)
            audio_thread.daemon = True
            audio_thread.start()
        except Exception as exc:
            self.notas.text = str(exc)

    def audio_stream(self):
        stream = sd.InputStream(channels=1, callback=self.audio_callback, blocksize=self.WINDOW_STEP, samplerate=self.SAMPLE_FREQ)
        with stream:
            while True:
                time.sleep(0.1)

    def audio_callback(self, indata, frames, time, status):
        # Convert indata to a NumPy array if needed
        indata_np = np.array(indata) if isinstance(indata, np.ndarray) else indata
        self.callback(indata_np, frames, time, status)

class MainApp(App):
    def build(self):
        tuner = TunerWidget()
        tuner.start_stream()
        return tuner
    def cambiar_personaje(self, instance, value):
        personaje_widget = self.root.ids._personaje
        circulo_widget = self.root.ids._circulo
        if value:
            personaje_widget.source = 'rositaaislada.png'
        else:
            personaje_widget.source = 'grelaaislado.png'

    
if __name__ == '__main__':
    MainApp().run()
