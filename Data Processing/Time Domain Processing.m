Fs = 100;                                           % Sampling frequency
T = 1/Fs;                                           % Sampling period
L = 1001;                                           % Length of signal
L1 = length(1:L/2)
L2 = length(1:L1/2)
t = (0:L-1)*T;                                      % Time vector
t1 = (0:L1-1)*T; 
t2 = (0:L2-1)*T;

A = c1
B = c3
C = c5
R = reference

%RAW SIGNALS FOR FULL CYCLE
figure(1)
plot(t,A,'-r','LineWidth',1.5)
hold on
plot(t,B, '-b', 'LineWidth',1.5)
plot(t,C, '-g', 'LineWidth',1.5)
grid on
xlabel('Time(ms)')
ylabel('Amplitude (mV)')
title('PEC Raw Signal')
legend('Crack Depth: 0mm', 'Crack Depth: 1mm', 'Crack Depth: 3mm' )

%TRUNCATED NORMALIZE SIGNAL WITH DC OFFSET CORRECTED
A1 = A(1:L/2)-mean(A(1:L/2))
B1 = B(1:L/2)-mean(B(1:L/2))
C1 = C(1:L/2)-mean(C(1:L/2))
R1 = R(1:L/2)-mean(R(1:L/2))

D = (A1-min(A1))./(max(A1)-min(A1))
E = (B1-min(B1))./(max(B1)-min(B1))
F = (C1-min(C1))./(max(C1)-min(C1))
RF = (R1-min(R1))./(max(R1)-min(R1))


figure(2)
plot(t1,D,'-r','LineWidth',1.5)
hold on
plot(t1,E, '-b','LineWidth',1.5)
plot(t1,F, '-g','LineWidth',1.5)
grid on
xlabel('Time(ms)')
ylabel('Amplitude (mV)')
title('Truncated Normalized Signal with Offset Corrected')
legend('Crack Depth: 0mm', 'Crack Depth: 1mm', 'Crack Depth: 3mm' )


%FFT
Fn = Fs/2;              % Nyquist Frequency
Fc=0.3;
Fv = linspace(0, 1, fix(L1/2)+1)*Fn;                 % Frequency Vector
Iv = 1:length(Fv);                                  % Index Vector
X = fft(D)/L1;
Y = fft(E)/L1;
Z = fft(F)/L1;
Ref = fft(RF)/L1;

figure(3)
plot(Fv, abs(X(Iv))*2, '-r','LineWidth',1.5)
hold on
plot(Fv, abs(Y(Iv))*2, '-b','LineWidth',1.5)
plot(Fv, abs(Z(Iv))*2, '-g','LineWidth',1.5)
grid on
xlabel('Frequency (Hz)')
ylabel('Amplitude (mV)')
title('Fast Fourier Transform of Normalized Signal')
legend('Crack Depth: 0mm','Crack Depth: 1mm', 'Crack Depth: 3mm' )

%Butterworth Low Pass Filter
[b,a] = butter(4,Fc/Fn);                               % Butterworth Transfer Function Coefficients
[SOS,G] = tf2sos(b,a);                              % Convert to Second-Order-Section For Stability

Refer = filtfilt(SOS,G,Ref);                              % Filter ‘X’ To Recover ‘S
c0mm = filtfilt(SOS,G,X);
c1mm = filtfilt(SOS,G,Y);
c3mm = filtfilt(SOS,G,Z);

real_c0mm =real(c0mm) - real(Refer);
real_c1mm =real(c1mm)-real(Refer);
real_c3mm =real(c3mm)-real(Refer);

% Find local maxima
[pks1,locs1] = findpeaks(real_c0mm(1:450));
[pks2,locs2] = findpeaks(real_c1mm(1:450));
[pks3,locs3] = findpeaks(real_c3mm(1:450));

% Find local minima
[trghs1,locst1] = findpeaks(-real_c0mm(1:450));
[trghs2,locst2] = findpeaks(-real_c1mm(1:450));
[trghs3,locst3] = findpeaks(-real_c3mm(1:450));

% Print the values of the peaks and valleys
maxima_minima=[pks1,  -trghs1; pks2,  -trghs2; pks3, -trghs3]*1000

figure(6)
plot(t1, real_c0mm, '-r','LineWidth',1.5)                   % Plot ‘S’
hold on
plot(t1, real_c1mm, '-b','LineWidth',1.5)                   % Plot ‘S’
plot(t1, real_c3mm, '-g','LineWidth',1.5)                   % Plot ‘S’
hold on
plot(t1(locs1),pks1,'rv','MarkerFaceColor','r');
plot(t1(locs2),pks2,'rv','MarkerFaceColor','r');
plot(t1(locs3),pks3,'rv','MarkerFaceColor','r');
plot(t1(locst1),-trghs1,'gv','MarkerFaceColor','g');
plot(t1(locst2),-trghs2,'gv','MarkerFaceColor','g');
plot(t1(locst3),-trghs3,'gv','MarkerFaceColor','g');
hold off;
grid on
legend('Crack Depth: 0mm', 'Crack Depth: 1mm', 'Crack Depth: 3mm')
title('Differential PEC Signal')
xlabel('Time (ms)')
ylabel('Amplitude (mV)')

