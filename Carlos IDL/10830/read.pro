; Date 23 - May - 2025

path = '/Users/carlos/OneDrive/Latex/AR14072/Scan_20250427/level3_20250429/'
stokes = readfits(path+'27Apr25ARM1-000.fits')

; % READFITS: Now reading 872 by 88 by 48 by 4 by 50 array
; stop
; The dimension 872 corresponds to the wavelength range,
; 88 and 48 are the spatial points, 4 the Stokes profiles,
; and the last one is the time domain (50).

; Let's start rotating the FOV

dimx = 88 & dimy = 48 & lam = 872 & time = 50

stokesr = dblarr(dimx,dimy,lam,4,time)

for tt = 0, time-1 do begin
  for ll = 0, lam-1 do begin

  stokesr[*,*,ll,0,tt] = rotate(reform(stokes[ll,*,*,0,tt]),90)
  stokesr[*,*,ll,1,tt] = rotate(reform(stokes[ll,*,*,1,tt]),90)
  stokesr[*,*,ll,2,tt] = rotate(reform(stokes[ll,*,*,2,tt]),90)
  stokesr[*,*,ll,3,tt] = rotate(reform(stokes[ll,*,*,3,tt]),90)    
  ; print,ll
endfor
  print,tt
endfor


; tvframe,stokes[0,*,*,0,0],/aspect
; pause
; tvframe,stokesr[*,*,0,0,0],/aspect
; stop

; -------------------- METHOD 1 ------------------------------------
; Let's make a frebin of the spectral domain, let's move from
; 1000 to 333 spectral points.

; stokesrf = dblarr(dimx,dimy,333,4)

; for xx = 0, dimx-1 do begin

;    stokesrf[xx,*,*,0] = frebin(reform(stokesr[xx,*,*,0]),466,333)
;    stokesrf[xx,*,*,1] = frebin(reform(stokesr[xx,*,*,1]),466,333)
;    stokesrf[xx,*,*,2] = frebin(reform(stokesr[xx,*,*,2]),466,333)
;    stokesrf[xx,*,*,3] = frebin(reform(stokesr[xx,*,*,3]),466,333)

; endfor


; cont = mean(stokesrf[200:240,350:450,332,0])
; stokesrf = stokesrf/cont
; ------------------------------------------------------------------

; -------------------- METHOD 2 ------------------------------------

wavep = 872

binning = 3
ori_nstok    = wavep
new_nstok    = wavep/binning

; From Manolo
; binning=3
; original_nstok=n_elements(original_Stokes)
; nuevo_nstok=n_elements(original_Stokes)/binning
; intermedio_Stokes=original_Stokes[0:nuevo_nstok*binning-1]
; intermedio_Stokes=reform(intermedio_Stokes,binning,nuevo_nstok)
; nuevo_Stokes = total(intermedio_Stokes,1)/binning

stokesrf2 = dblarr(dimx,dimy,new_nstok,4,50)
for time = 0, 49 do begin

for xx = 0, dimx-1 do begin
  for yy = 0, dimy-1 do begin

   ori_Stokes = reform(stokesr[xx,yy,*,0,time])
   inter_Stokes = ori_Stokes[0:new_nstok*binning-1]   
   inter_Stokes = reform(inter_Stokes,binning,new_nstok) 
   stokesrf2[xx,yy,*,0,time]   = total(inter_Stokes,1)/binning

   ori_Stokes = reform(stokesr[xx,yy,*,1,time])
   inter_Stokes = ori_Stokes[0:new_nstok*binning-1]   
   inter_Stokes = reform(inter_Stokes,binning,new_nstok) 
   stokesrf2[xx,yy,*,1,time]   = total(inter_Stokes,1)/binning

   ori_Stokes = reform(stokesr[xx,yy,*,2,time])
   inter_Stokes = ori_Stokes[0:new_nstok*binning-1]   
   inter_Stokes = reform(inter_Stokes,binning,new_nstok) 
   stokesrf2[xx,yy,*,2,time]   = total(inter_Stokes,1)/binning

   ori_Stokes = reform(stokesr[xx,yy,*,3,time])
   inter_Stokes = ori_Stokes[0:new_nstok*binning-1]   
   inter_Stokes = reform(inter_Stokes,binning,new_nstok) 
   stokesrf2[xx,yy,*,3,time]   = total(inter_Stokes,1)/binning      

  endfor
  ; print,xx
endfor  
  print,time
endfor

; Original
; cont = mean(stokesrf2[*,*,0:10,0,0:49])

; New
cont = 1e5 ; This comes from the data reduction. Manolo says that the data
; is normalised by the flatfield average continuum and then multiplied by
; 1e5
stokesrf2 = stokesrf2/cont

stokes_He = stokesrf2

writefits,'He_10830.fits',stokes_He

stop

; FF1WLOFF=        8540.37725722 / WL-offset FF1                                  
; FF1WLDSP=            0.0111461 / WL-dispersion FF1                              
; FF1FWHMA=             0.130983 / spectral FWHM [A] FF1                          
; FF1FWHMP=              11.7514 / spectral FWHM [pix] FF1                        
; FF1STRAY=           0.00200000 / spectral straylight FF1   

ll = 333
l0 = 8542.091d0 ; From LINES_NLTE
l0n = 8540.37725722d0 ;[A]
ini = l0n-l0

; We did a binning of 3 so we need to multiply the sampling by 3

wave = (dindgen(ll)*0.0111461d0*3.+ini)*1e3

print,'initial, step, final'
print,wave[0],0.0111461d0*3.*1e3,wave[332] 
      ;       -1713.7428       33.438300       9387.7728


end