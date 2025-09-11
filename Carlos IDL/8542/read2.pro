; Date 23 - May - 2025

; We created an array in Ca_8542.fits containing the original data, rotated,
; and interpolated along the spectral domain.

stokes = readfits('Ca_8542.fits')
; % READFITS: Now reading 92 by 48 by 333 by 4 by 50 array

xdim = 92 & ydim = 48 & nwave = 333 & time = 50

; Now, we know that the pixel scale along the slit is slightly different between
; 10830 and 8542 so we are going to re-scale it. In the case of the 10830 channel,
; the original data is 

; IDL> s=readfits('He_10830.fits')
; % READFITS: Now reading 88 by 48 by 290 by 4 by 50 array

; Let's first match both scales

xdim1 = 88 

stokesn = dblarr(xdim1,ydim,nwave,4,time)

for ii = 0, time-1 do begin
 for jj = 0, nwave-1 do begin

   stokesn[*,*,jj,0,ii] = congrid(reform(stokes[*,*,jj,0,ii]),xdim1,ydim)
   stokesn[*,*,jj,1,ii] = congrid(reform(stokes[*,*,jj,1,ii]),xdim1,ydim)
   stokesn[*,*,jj,2,ii] = congrid(reform(stokes[*,*,jj,2,ii]),xdim1,ydim)
   stokesn[*,*,jj,3,ii] = congrid(reform(stokes[*,*,jj,3,ii]),xdim1,ydim)      

 endfor
endfor 


; Now, we have that with double sampling the step along the Y-axis corresponds
; to 0.1875 arcsec/pix while the step along the slit (horizontal axis) corresponds to
; 0.135 arcsec/pix

; So we can interpolate to a uniform grid of 0.135 x 0.135

; First, let's create an array with the right dimension. 

yscale = 0.1875/0.135d0
ydim1 = ydim*yscale

stokesn2 = dblarr(xdim1,ydim1,nwave,4,time)

for ii = 0, time-1 do begin
 for jj = 0, nwave-1 do begin

   stokesn2[*,*,jj,0,ii] = congrid(reform(stokesn[*,*,jj,0,ii]),xdim1,ydim1)
   stokesn2[*,*,jj,1,ii] = congrid(reform(stokesn[*,*,jj,1,ii]),xdim1,ydim1)
   stokesn2[*,*,jj,2,ii] = congrid(reform(stokesn[*,*,jj,2,ii]),xdim1,ydim1)
   stokesn2[*,*,jj,3,ii] = congrid(reform(stokesn[*,*,jj,3,ii]),xdim1,ydim1)      

 endfor
endfor 

writefits,'Ca_8542_2.fits',stokesn2


end