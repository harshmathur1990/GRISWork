; Date 10 - Jun - 2025

; Let's re-normalise the rebinned data because the Ca II data does not reach
; the continuum

; This is the wavelength information for Channel 2

ll = 333
l0n = 8540.67304823 ;[A]
cawave = (dindgen(ll)*0.0109907d0*3.+l0n)

; IDL> print,cawave[332]
;        8551.6196

; BASS2000 says that 8551.48 has an intensity of 9744 counts
; So we need to re-normalise to 9744/10000

scale = 9744d0/10000d0

s = readfits('Ca_8542_3.fits')

s = s*scale

writefits,'Ca_8542_4.fits',s







end