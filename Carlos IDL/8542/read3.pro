; Date 9 - June - 2025

; The next correction is due to the impact of atmospheric refraction. From the plot made
; in /imasrep/Continuum2D we have



; tvframebar,hecont[inix:finx-4,iniy:finy-3]
; tvframebar,cacont[inix+4:finx,iniy+3:finy]

; So, we need to reduce the maps accordingly

stokes = readfits('Ca_8542_2.fits')
stokesn = stokes[4:*,3:*,*,*,*]

writefits,'Ca_8542_3.fits',stokesn


end