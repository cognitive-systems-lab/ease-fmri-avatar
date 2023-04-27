more off;
pkg load instrument-control;
pp = parallel("/dev/parport0", 0);
pulse = 0;
stat = 63;
oldstat = 63;
t = zeros(2000,1);
t1 = 0;
t2 = 0;
t3 = 0;
TR = 0;
PW = 0;
t_wait = 0;
fprintf('Waiting for pulses\n');
while pulse <3
  oldstat = stat;
  stat = pp_stat (pp);
  if ((stat == 127) && (stat != oldstat))
     if (t(1) > 0)
      t(3) = time;
      TR = t(3)-t(1);
      fprintf('TR = %f sec (%f msec)\n',TR, round(TR * 1000.0));
    endif

    if (t(1) == 0)
      t(1) = time;
      fprintf('pulse detected, starting\n');
      
    endif
    pulse = pulse + 1;
  endif
  if ((stat == 63) && (stat != oldstat))
    t(2) = time;
    PW = t(2)-t(1);
    fprintf('pulsewidth = %f sec (%f msec) \n',PW,round(PW * 1000.0));
    pulse = pulse + 1;
  endif
endwhile
if (pulse > 2)
  while (t_wait < (TR + 1))
    oldstat = stat;
    stat = pp_stat (pp);
    if (stat != oldstat)
      t(pulse) = time;
      pulse = pulse + 1;
      if (mod(pulse,2) == 0)
         fprintf('.');
      endif
      if (mod(pulse,100) == 0)
         fprintf('\n');
      endif        
    endif
    t_wait = time - t(pulse-1);
  endwhile
endif
fprintf('\n');
fprintf('finished\n');
fprintf('number of detected pulses is %d \n', (pulse/2));
pp_close(pp);
pkg unload instrument-control;
more on;
  
