; Date 9 - June - 2025

; We created an array in He_10830.fits containing the original data, rotated,
; and interpolated along the spectral domain.

stokes = readfits('He_10830.fits')
; % READFITS: Now reading 88 by 48 by 290 by 4 by 50 array

xdim = 88 & ydim = 48 & nwave = 290 & time = 50


; Now, we have that with double sampling the step along the Y-axis corresponds
; to 0.1875 arcsec/pix while the step along the slit (horizontal axis) corresponds to
; 0.135 arcsec/pix

; So we can interpolate to a uniform grid of 0.135 x 0.135

; First, let's create an array with the right dimension. 

yscale = 0.1875/0.135d0
ydim1 = ydim*yscale

stokesn = dblarr(xdim,ydim1,nwave,4,time)

for ii = 0, time-1 do begin
 for jj = 0, nwave-1 do begin

   stokesn[*,*,jj,0,ii] = congrid(reform(stokes[*,*,jj,0,ii]),xdim,ydim1)
   stokesn[*,*,jj,1,ii] = congrid(reform(stokes[*,*,jj,1,ii]),xdim,ydim1)
   stokesn[*,*,jj,2,ii] = congrid(reform(stokes[*,*,jj,2,ii]),xdim,ydim1)
   stokesn[*,*,jj,3,ii] = congrid(reform(stokes[*,*,jj,3,ii]),xdim,ydim1)      

 endfor
endfor 

writefits,'He_10830_2.fits',stokesn


end