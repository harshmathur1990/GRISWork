; Date 9 - June - 2025

; The next correction is due to the impact of atmospheric refraction. From the plot made
; in /imasrep/Continuum2D we have



; tvframebar,hecont[inix:finx-4,iniy:finy-3]
; tvframebar,cacont[inix+4:finx,iniy+3:finy]

; So, we need to reduce the maps accordingly

dimx = 88 & dimy =66
finx = dimx-4-1. & finy = dimy-3-1.

stokes = readfits('He_10830_2.fits')
stokesn = stokes[0:finx,0:finy,*,*,*]

writefits,'He_10830_3.fits',stokesn

; STOKESN         DOUBLE    = Array[84, 63, 333, 4, 50]
end