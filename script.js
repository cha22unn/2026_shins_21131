/**
 * AI 층간소음 선택적 노이즈 캔슬링 시뮬레이터 (Selective ANC Simulator)
 * - Pure Client-side Web Audio & SVM-RBF Spectral Classifier
 * - GitHub Pages Compatible (Relative paths, Zero backend)
 */

(function () {
  'use strict';

  // ==========================================
  // 1. 전역 상태 및 DOM 요소 참조
  // ==========================================

  // 로컬 file:/// 직접 실행 시 CORS 차단 방지를 위한 내장 기본 SVM 모델 (Fallback)
  const DEFAULT_SVM_MODEL = {
    model_type: "SVC_RBF",
    features: [
      "Low_Freq_Energy (20-250Hz)",
      "Mid_High_Freq_Energy (250-4000Hz)",
      "Low_Freq_Ratio (%)"
    ],
    classes: [
      "Footsteps",
      "Voice"
    ],
    scaler_mean: [
      13.085708333333333,
      32.108491666666666,
      29.739333333333338
    ],
    scaler_scale: [
      9.4827344314688,
      11.90784696768328,
      21.017065594945034
    ],
    support_vectors: [
      [-1.0853734656090266, 0.6558623363676618, -1.1038331316296208],
      [0.20649019339492075, 0.17441510114883568, 0.039047633122677464],
      [0.25472522552680277, -0.1074158636853492, 0.1770307396081757],
      [0.9513175478931316, -0.4722089292822549, 0.7494227296152595],
      [-0.006844896248273355, 0.20154007184056447, -0.1113063725649688],
      [0.6138094142287507, -0.32194666903856983, 0.491537061632018],
      [1.9353164215760996, -1.1855200780610289, 1.6111034394264219],
      [0.3621942269382604, 0.01034681867072242, 0.1974903036732668],
      [0.3734251646777563, 0.07790731068773758, 0.17798234723911],
      [-1.1447445261473945, 1.0708575923036248, -1.1894778184137231],
      [-0.007551443505124308, 0.1720469148531493, -0.10226610007109127],
      [-1.2981496447356073, -2.46919461985554, -0.354917926084193],
      [-0.8141964102370672, 0.9614591423961478, -0.8930520413776357],
      [-1.1518416351602403, 1.1669790828725184, -1.2013729138004041],
      [-1.139503421868963, 1.1576994876358742, -1.1899536222291904],
      [1.0958539165856485, -1.307683220058731, 1.376532158401075],
      [0.30234861973484345, -0.05882605550505727, 0.18749842354845483],
      [-1.1581161966203342, 1.1053222609472326, -1.2037519328777402],
      [-1.1578947415099778, 0.8248769370307368, -1.1875746031518541],
      [-1.2090930539280553, 0.6060296502733221, -1.2265905160201676],
      [0.3873452001858641, 0.08156036401619035, 0.18511940447111877],
      [-0.02543658003675334, -0.0012085868004288196, -0.05516152233983508],
      [-0.7107136008120221, 0.8448133705979703, -0.7926574363140492],
      [0.5489125214129218, -0.21944283242456325, 0.406368178663383],
      [0.10340811226480603, 0.11883830361389429, -0.008532748424046032],
      [-0.5557055690440877, 0.4564224202816354, -0.5951988528951466],
      [0.3349130664386589, 0.10126165851860389, 0.14562768778733803],
      [-0.7885708903244146, 0.9752483689830886, -0.8740198887589463],
      [-0.04292098827345219, 0.40076164450925794, -0.19790266698000558],
      [-0.8123509509840985, 0.9963688956981653, -0.8959068642704392],
      [1.4074096204125655, -1.024785731609113, 1.2989761364799157],
      [-0.37030545975011864, 0.4319763553641052, -0.4419900243146971],
      [1.0533134444343564, -0.5349490704704609, 0.8341158087684273],
      [-0.8121189503922966, 0.8064773052085767, -0.871165065866143],
      [-0.4400743649937851, 0.35042508899030406, -0.4767237028438052],
      [0.11553541592717251, 0.26447336297487267, -0.05040348418516266],
      [1.3640360552270756, -0.7514617622271638, 1.1024691606919474],
      [-0.038544613473554624, 0.32529040252580105, -0.17220926094477476],
      [0.3349130664386589, -0.03909116970766953, 0.20034512656607031],
      [-0.8133633172028699, 0.9837133753166064, -0.895431060454972],
      [-0.8555768794293507, 1.0750984932941272, -0.9406324229243591],
      [-0.8585401597155463, 0.6836591329588763, -0.8944794528240374]
    ],
    dual_coef: [
      -1.0, -1.0, -1.0, -0.10277938249613511, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      -1.0, -0.009683767352702925, -1.0, -1.0, -1.0, -0.8705902303588159, -1.0, -1.0, -1.0, -1.0,
      -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 0.9830533802076539, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0
    ],
    intercept: -0.9490051849054875,
    gamma: 0.3333333333333333
  };

  let svmModel = null;
  let audioContext = null;
  let currentAudioBuffer = null;
  let currentMonoData = null;
  let currentAnalysis = null;

  // 오디오 재생 관련 상태
  let activeSourceNode = null;
  let playbackStartTime = 0;
  let playbackDuration = 0;
  let playbackTimerId = null;
  let isPlayingType = null; // 'orig' | 'anc' | null

  // DOM 캐싱
  const modelStatusBadge = document.getElementById('model-status-badge');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const btnSampleFootstep = document.getElementById('btn-sample-footstep');
  const btnSampleVoice = document.getElementById('btn-sample-voice');

  const audioMeta = document.getElementById('audio-meta');
  const metaFilename = document.getElementById('meta-filename');
  const metaSr = document.getElementById('meta-sr');
  const metaLength = document.getElementById('meta-length');
  const metaChannels = document.getElementById('meta-channels');

  const decisionBadge = document.getElementById('decision-badge');
  const badgeIcon = document.getElementById('badge-icon');
  const badgeTitle = document.getElementById('badge-title');
  const badgeDesc = document.getElementById('badge-desc');

  const scoreValue = document.getElementById('score-value');
  const scorePointer = document.getElementById('score-pointer');

  const valLowEnergy = document.getElementById('val-low-energy');
  const barLowEnergy = document.getElementById('bar-low-energy');
  const valMidEnergy = document.getElementById('val-mid-energy');
  const barMidEnergy = document.getElementById('bar-mid-energy');
  const valRatio = document.getElementById('val-ratio');
  const barRatio = document.getElementById('bar-ratio');

  const btnPlayOrig = document.getElementById('btn-play-orig');
  const btnPlayAnc = document.getElementById('btn-play-anc');
  const btnStop = document.getElementById('btn-stop');
  const playbackStatus = document.getElementById('playback-status');
  const playbackTime = document.getElementById('playback-time');

  const canvasOrig = document.getElementById('canvas-orig');
  const canvasAnti = document.getElementById('canvas-anti');
  const canvasOut = document.getElementById('canvas-out');

  const playheadOrig = document.getElementById('playhead-orig');
  const playheadAnti = document.getElementById('playhead-anti');
  const playheadOut = document.getElementById('playhead-out');

  const tagAnti = document.getElementById('tag-anti');
  const titleAnti = document.getElementById('title-anti');
  const infoAnti = document.getElementById('info-anti');

  const tagOut = document.getElementById('tag-out');
  const titleOut = document.getElementById('title-out');
  const infoOut = document.getElementById('info-out');

  // ==========================================
  // 2. SVM 모델 로드 (내장 모델 Fallback 지원)
  // ==========================================
  async function loadSVMModel() {
    // 1) 웹 서버 환경(http://, https://)인 경우 원격/로컬 JSON 파일 fetch 시도
    if (window.location.protocol !== 'file:') {
      try {
        const res = await fetch('./svm_model_export.json');
        if (res.ok) {
          svmModel = await res.json();
          console.log('SVM Model Loaded via fetch:', svmModel);
          applyModelReadyState('외부 JSON 로드');
          return;
        }
      } catch (err) {
        console.warn('fetch(./svm_model_export.json) 실패 또는 차단됨, 내장 모델로 자동 전환합니다:', err);
      }
    }

    // 2) 로컬 직접 실행(file://)이거나 fetch 실패 시: 내장 DEFAULT_SVM_MODEL 즉시 적용
    svmModel = DEFAULT_SVM_MODEL;
    console.log('SVM Model Loaded via Embedded Fallback (CORS Safe):', svmModel);
    applyModelReadyState('내장 모델 로드 (로컬 호환)');
  }

  function applyModelReadyState(sourceLabel) {
    modelStatusBadge.classList.add('loaded');
    modelStatusBadge.innerHTML = `
      <span class="status-dot"></span> SVM 모델 준비 완료 (${svmModel.support_vectors.length} SVs, RBF γ=${svmModel.gamma.toFixed(3)})
    `;
  }

  // ==========================================
  // 3. Web Audio Context 초기화
  // ==========================================
  function getAudioContext() {
    if (!audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioCtx();
    }
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
    return audioContext;
  }

  // ==========================================
  // 4. 고성능 Radix-2 In-place FFT (Double Precision)
  // ==========================================
  function radix2FFT(real, imag) {
    const n = real.length;

    // 1) Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        let temp = real[i];
        real[i] = real[j];
        real[j] = temp;

        temp = imag[i];
        imag[i] = imag[j];
        imag[j] = temp;
      }
      let k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // 2) Butterfly computations
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const angle = (-2.0 * Math.PI) / len;
      const wStepReal = Math.cos(angle);
      const wStepImag = Math.sin(angle);

      for (let i = 0; i < n; i += len) {
        let wr = 1.0;
        let wi = 0.0;
        for (let k = 0; k < half; k++) {
          const pos = i + k;
          const match = pos + half;

          const ur = real[pos];
          const ui = imag[pos];
          const vr = real[match] * wr - imag[match] * wi;
          const vi = real[match] * wi + imag[match] * wr;

          real[pos] = ur + vr;
          imag[pos] = ui + vi;
          real[match] = ur - vr;
          imag[match] = ui - vi;

          const nextWr = wr * wStepReal - wi * wStepImag;
          wi = wr * wStepImag + wi * wStepReal;
          wr = nextWr;
        }
      }
    }
  }

  // ==========================================
  // 5. 특징 추출 및 SVM 예측 (Python 100% 일치)
  // ==========================================
  function extractFeaturesAndPredict(monoSamples, sampleRate) {
    if (!svmModel) {
      throw new Error('SVM 모델이 아직 로드되지 않았습니다.');
    }

    const N = monoSamples.length;
    if (N === 0) throw new Error('오디오 샘플이 비어 있습니다.');

    // 1) RMS 음량 정규화
    let sumSq = 0.0;
    for (let i = 0; i < N; i++) {
      sumSq += monoSamples[i] * monoSamples[i];
    }
    const rms = Math.sqrt(sumSq / N);

    const normSamples = new Float64Array(N);
    if (rms > 0) {
      for (let i = 0; i < N; i++) {
        normSamples[i] = monoSamples[i] / rms;
      }
    } else {
      normSamples.set(monoSamples);
    }

    // 2) N보다 크거나 같은 가장 작은 2의 거듭제곱 M
    let M = 1;
    while (M < N) {
      M <<= 1;
    }

    // Radix-2 FFT 입력 버퍼 준비 (zero-padding)
    const real = new Float64Array(M);
    const imag = new Float64Array(M);
    real.set(normSamples); // 나머지 [N...M-1]은 자동으로 0.0

    // FFT 수행
    radix2FFT(real, imag);

    // 3) 양의 주파수 대역 파워 스펙트럼 계산 및 핵심 속성 추출
    const kMax = M >> 1;
    const normFactor = (N * M);
    let lowFreqEnergy = 0.0;
    let midHighFreqEnergy = 0.0;

    for (let k = 0; k <= kMax; k++) {
      const freq = (k * sampleRate) / M;
      if (freq >= 20 && freq <= 250) {
        const power = ((real[k] * real[k] + imag[k] * imag[k]) / normFactor) * 100.0;
        lowFreqEnergy += power;
      } else if (freq > 250 && freq <= 4000) {
        const power = ((real[k] * real[k] + imag[k] * imag[k]) / normFactor) * 100.0;
        midHighFreqEnergy += power;
      }
    }

    const totalTargetEnergy = lowFreqEnergy + midHighFreqEnergy;
    const lowFreqRatio = totalTargetEnergy > 0 ? (lowFreqEnergy / totalTargetEnergy) * 100.0 : 0.0;

    const rawFeatures = [lowFreqEnergy, midHighFreqEnergy, lowFreqRatio];

    // 4) SVM 표준화 (StandardScaler: (x - mean) / scale)
    const scalerMean = svmModel.scaler_mean;
    const scalerScale = svmModel.scaler_scale;
    const scaledFeatures = [
      (rawFeatures[0] - scalerMean[0]) / scalerScale[0],
      (rawFeatures[1] - scalerMean[1]) / scalerScale[1],
      (rawFeatures[2] - scalerMean[2]) / scalerScale[2]
    ];

    // 5) RBF 커널 결정 함수 계산
    // decision = sum(dual_coef[i] * exp(-gamma * ||x_scaled - sv[i]||^2)) + intercept
    const svs = svmModel.support_vectors;
    const dualCoef = svmModel.dual_coef;
    const gamma = svmModel.gamma;
    const intercept = svmModel.intercept;

    let decisionScore = intercept;
    const numSVs = svs.length;

    for (let i = 0; i < numSVs; i++) {
      const sv = svs[i];
      const d0 = scaledFeatures[0] - sv[0];
      const d1 = scaledFeatures[1] - sv[1];
      const d2 = scaledFeatures[2] - sv[2];
      const distSq = d0 * d0 + d1 * d1 + d2 * d2;

      decisionScore += dualCoef[i] * Math.exp(-gamma * distSq);
    }

    // 6) 판별 결과
    // decision > 0: classes[1] ("Voice"), decision <= 0: classes[0] ("Footsteps")
    const isVoice = decisionScore > 0;
    const predictionClass = isVoice ? svmModel.classes[1] : svmModel.classes[0];

    return {
      N,
      M,
      sampleRate,
      rawFeatures: {
        lowFreqEnergy,
        midHighFreqEnergy,
        lowFreqRatio
      },
      scaledFeatures,
      decisionScore,
      predictionClass,
      isVoice
    };
  }

  // ==========================================
  // 6. UI 및 파형 업데이트
  // ==========================================
  function updateDashboard(filename, analysis, monoSamples) {
    currentAnalysis = analysis;
    currentMonoData = monoSamples;

    // 메타데이터
    audioMeta.style.display = 'grid';
    metaFilename.textContent = filename;
    metaSr.textContent = `${analysis.sampleRate.toLocaleString()} Hz`;
    const durSec = (analysis.N / analysis.sampleRate).toFixed(2);
    metaLength.textContent = `${durSec}초 (${analysis.N.toLocaleString()} 샘플)`;
    metaChannels.textContent = currentAudioBuffer.numberOfChannels === 1 ? '1 (Mono)' : `${currentAudioBuffer.numberOfChannels} (Stereo → Mono 변환)`;

    // 핵심 속성 수치 및 막대 표시
    const lowE = analysis.rawFeatures.lowFreqEnergy;
    const midE = analysis.rawFeatures.midHighFreqEnergy;
    const ratio = analysis.rawFeatures.lowFreqRatio;

    valLowEnergy.textContent = lowE.toFixed(4);
    valMidEnergy.textContent = midE.toFixed(4);
    valRatio.textContent = `${ratio.toFixed(2)}%`;

    // 프로그레스 바 (최대 스케일 기준 대략화)
    barLowEnergy.style.width = `${Math.min(100, (lowE / 35.0) * 100)}%`;
    barMidEnergy.style.width = `${Math.min(100, (midE / 55.0) * 100)}%`;
    barRatio.style.width = `${Math.min(100, Math.max(0, ratio))}%`;

    // SVM 결정 점수
    const score = analysis.decisionScore;
    const scoreSign = score >= 0 ? `+${score.toFixed(4)}` : score.toFixed(4);
    scoreValue.textContent = scoreSign;

    // 점수 포인터 위치 (-2.0 ~ +2.0 범위를 0% ~ 100%에 매핑)
    const pointerPercent = Math.min(95, Math.max(5, ((score - (-2.0)) / 4.0) * 100));
    scorePointer.style.left = `${pointerPercent}%`;

    // AI 판별 배지 및 테마
    decisionBadge.className = 'decision-badge';
    if (!analysis.isVoice) {
      // Footsteps (층간소음 발소리)
      decisionBadge.classList.add('badge-footsteps');
      badgeIcon.textContent = '⚠️';
      badgeTitle.textContent = '층간소음(발소리) 감지 → 역위상 파동 생성 및 상쇄 작동';
      badgeDesc.textContent = `저주파 충격음 에너지 특성이 지배적입니다 (점수: ${scoreSign} ≤ 0). 원본과 반대 위상(-Signal)의 ANC 제어 음파를 합성하여 소음을 완전히 상쇄(0dB)시킵니다.`;

      // 그래프 2 라벨 업데이트
      tagAnti.className = 'wave-tag tag-anti';
      titleAnti.textContent = 'AI 제어 파형: 역위상 상쇄 파동 (-원본 신호)';
      infoAnti.textContent = 'c(t) = -x(t) [위상 180° 반전 생성]';

      // 그래프 3 라벨 업데이트
      tagOut.className = 'wave-tag tag-out';
      titleOut.textContent = '최종 출력 파형: 소음 상쇄 소멸 (Silent 0)';
      infoOut.textContent = 'y(t) = x(t) + c(t) = 0 [소음 물리적 소멸]';
    } else {
      // Voice (인간 말소리)
      decisionBadge.classList.add('badge-voice');
      badgeIcon.textContent = '🛡️';
      badgeTitle.textContent = '말소리 감지 → 상쇄 미작동 (원본 음성 투과)';
      badgeDesc.textContent = `인간 발성 주파수 스펙트럼(250~4000Hz)이 확인되었습니다 (점수: ${scoreSign} > 0). 대화와 음성을 보존하기 위해 제어 파동 생성을 우회(Bypass)하여 원본 그대로 전달합니다.`;

      // 그래프 2 라벨 업데이트
      tagAnti.className = 'wave-tag tag-orig';
      titleAnti.textContent = 'AI 제어 파형: 제어 파동 미생성 (무음 0)';
      infoAnti.textContent = 'c(t) = 0 [ANC 시스템 바이패스]';

      // 그래프 3 라벨 업데이트
      tagOut.className = 'wave-tag tag-out';
      titleOut.textContent = '최종 출력 파형: 원본 음성 보존 투과 (Bypass)';
      infoOut.textContent = 'y(t) = x(t) + 0 = x(t) [원음 100% 투과]';
    }

    // 재생 버튼 활성화
    btnPlayOrig.disabled = false;
    btnPlayAnc.disabled = false;
    btnStop.disabled = false;
    stopAudioPlayback();

    // 3단 파형 그리기
    renderAllWaveforms();
  }

  // ==========================================
  // 7. 3단 파형 시각화 렌더러 (HTML5 Canvas)
  // ==========================================
  function setupHiDPICanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || 600;
    const height = rect.height || 110;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    const ctx = canvas.getContext('2d');
    ctx.resetTransform?.();
    ctx.scale(dpr, dpr);
    return { ctx, width, height };
  }

  function drawWaveform(canvas, data, strokeColor, isFlatZero = false) {
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    ctx.clearRect(0, 0, width, height);

    const midY = height / 2;

    // 배경 그리드 라인
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < width; x += 40) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
    }
    for (let y = 0; y < height; y += 25) {
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
    }
    ctx.stroke();

    // 중앙 축 (0 레벨 기준선)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    if (!data || data.length === 0) return;

    if (isFlatZero) {
      // 완벽한 0 직선 (상쇄 또는 무음)
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(width, midY);
      ctx.stroke();

      // 글로우 효과
      ctx.shadowColor = strokeColor;
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.shadowBlur = 0;
      return;
    }

    // 일반 파형 렌더링 (Peak Envelope 방식)
    const N = data.length;
    const step = N / width;
    const ampScale = (height / 2) * 0.9;

    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = strokeColor;
    ctx.lineWidth = 1.2;

    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const startIdx = Math.floor(x * step);
      const endIdx = Math.min(N, Math.floor((x + 1) * step));

      let minVal = 0.0;
      let maxVal = 0.0;
      for (let i = startIdx; i < endIdx; i++) {
        const val = data[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const yTop = midY - maxVal * ampScale;
      const yBottom = midY - minVal * ampScale;

      ctx.moveTo(x, Math.min(yTop, midY - 0.5));
      ctx.lineTo(x, Math.max(yBottom, midY + 0.5));
    }
    ctx.stroke();
  }

  function renderAllWaveforms() {
    if (!currentMonoData || !currentAnalysis) return;

    const isVoice = currentAnalysis.isVoice;

    // 그래프 1: 원본 입력 오디오 파형
    drawWaveform(canvasOrig, currentMonoData, '#38bdf8');

    if (!isVoice) {
      // Footsteps:
      // 그래프 2: AI 제어 파형 (-원본 신호)
      const antiSamples = new Float32Array(currentMonoData.length);
      for (let i = 0; i < currentMonoData.length; i++) {
        antiSamples[i] = -currentMonoData[i];
      }
      drawWaveform(canvasAnti, antiSamples, '#f43f5e');

      // 그래프 3: 최종 출력 파형 (완전 상쇄 = 0 직선)
      drawWaveform(canvasOut, null, '#10b981', true);
    } else {
      // Voice:
      // 그래프 2: AI 제어 파형 (미생성 = 0 직선)
      drawWaveform(canvasAnti, null, '#64748b', true);

      // 그래프 3: 최종 출력 파형 (원본 그대로 투과)
      drawWaveform(canvasOut, currentMonoData, '#10b981');
    }
  }

  // 윈도우 리사이즈 시 캔버스 다시 그리기
  window.addEventListener('resize', () => {
    if (currentMonoData && currentAnalysis) {
      renderAllWaveforms();
    }
  });

  // ==========================================
  // 8. 오디오 재생 제어 (Web Audio API)
  // ==========================================
  function stopAudioPlayback() {
    if (activeSourceNode) {
      try {
        activeSourceNode.stop();
      } catch (e) {
        // 이미 중지되었거나 재생 시작 전
      }
      activeSourceNode.disconnect();
      activeSourceNode = null;
    }

    if (playbackTimerId) {
      cancelAnimationFrame(playbackTimerId);
      playbackTimerId = null;
    }

    isPlayingType = null;
    btnPlayOrig.classList.remove('active');
    btnPlayAnc.classList.remove('active');

    playbackStatus.innerHTML = '<span class="dot-idle"></span> 준비됨';

    // 플레이헤드 숨김
    playheadOrig.style.display = 'none';
    playheadAnti.style.display = 'none';
    playheadOut.style.display = 'none';
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  }

  function startPlaybackProgress(duration) {
    playbackDuration = duration;
    playbackStartTime = audioContext.currentTime;

    playheadOrig.style.display = 'block';
    playheadAnti.style.display = 'block';
    playheadOut.style.display = 'block';

    function tick() {
      if (!activeSourceNode) return;

      const elapsed = audioContext.currentTime - playbackStartTime;
      const progress = Math.min(1.0, elapsed / playbackDuration);

      const percent = `${progress * 100}%`;
      playheadOrig.style.left = percent;
      playheadAnti.style.left = percent;
      playheadOut.style.left = percent;

      playbackTime.textContent = `${formatTime(elapsed)} / ${formatTime(playbackDuration)}`;

      if (elapsed < playbackDuration) {
        playbackTimerId = requestAnimationFrame(tick);
      } else {
        stopAudioPlayback();
      }
    }

    tick();
  }

  function playAudio(type) {
    if (!currentAudioBuffer || !currentAnalysis) return;

    const ctx = getAudioContext();
    stopAudioPlayback();

    isPlayingType = type;
    const duration = currentAudioBuffer.duration;

    if (type === 'orig') {
      // 1) 원본 소리 재생
      btnPlayOrig.classList.add('active');
      playbackStatus.innerHTML = '<span class="dot-playing"></span> 원본 소리 재생 중...';

      const source = ctx.createBufferSource();
      source.buffer = currentAudioBuffer;
      source.connect(ctx.destination);
      source.onended = () => {
        if (activeSourceNode === source) stopAudioPlayback();
      };
      source.start(0);
      activeSourceNode = source;
      startPlaybackProgress(duration);
    } else if (type === 'anc') {
      // 2) 노이즈 캔슬링 적용 후 소리 재생
      btnPlayAnc.classList.add('active');

      if (!currentAnalysis.isVoice) {
        // [Footsteps] -> 상쇄 작동 (완전 무음 버퍼 재생)
        playbackStatus.innerHTML = '<span class="dot-playing" style="background:#ef4444;box-shadow:0 0 8px #ef4444;"></span> ANC 작동 중: 발소리가 상쇄되어 소멸됨 (무음 출력)';

        // 완전 무음 버퍼 생성
        const silentBuffer = ctx.createBuffer(1, currentAudioBuffer.length, currentAudioBuffer.sampleRate);
        const source = ctx.createBufferSource();
        source.buffer = silentBuffer;
        source.connect(ctx.destination);
        source.onended = () => {
          if (activeSourceNode === source) stopAudioPlayback();
        };
        source.start(0);
        activeSourceNode = source;
        startPlaybackProgress(duration);
      } else {
        // [Voice] -> 상쇄 미작동 (원본 음성 투과)
        playbackStatus.innerHTML = '<span class="dot-playing" style="background:#10b981;box-shadow:0 0 8px #10b981;"></span> ANC 바이패스: 원본 말소리 그대로 투과 중';

        const source = ctx.createBufferSource();
        source.buffer = currentAudioBuffer;
        source.connect(ctx.destination);
        source.onended = () => {
          if (activeSourceNode === source) stopAudioPlayback();
        };
        source.start(0);
        activeSourceNode = source;
        startPlaybackProgress(duration);
      }
    }
  }

  // ==========================================
  // 9. 오디오 데이터 로드 및 분석 파이프라인
  // ==========================================
  async function processAudioArrayBuffer(arrayBuffer, filename) {
    try {
      decisionBadge.className = 'decision-badge badge-idle';
      badgeIcon.textContent = '🔄';
      badgeTitle.textContent = '오디오 디코딩 및 스펙트럼 분석 중...';
      badgeDesc.textContent = '고성능 Radix-2 FFT 변환 및 SVM-RBF 분류를 실행하고 있습니다.';

      // UI 갱신을 위한 짧은 대기
      await new Promise(r => setTimeout(r, 30));

      const ctx = getAudioContext();
      const decodedBuffer = await ctx.decodeAudioData(arrayBuffer);
      currentAudioBuffer = decodedBuffer;

      // 모노 채널 변환
      const N = decodedBuffer.length;
      const numChannels = decodedBuffer.numberOfChannels;
      const mono = new Float32Array(N);

      if (numChannels === 1) {
        mono.set(decodedBuffer.getChannelData(0));
      } else {
        for (let c = 0; c < numChannels; c++) {
          const chData = decodedBuffer.getChannelData(c);
          for (let i = 0; i < N; i++) {
            mono[i] += chData[i];
          }
        }
        for (let i = 0; i < N; i++) {
          mono[i] /= numChannels;
        }
      }

      // 특징 추출 및 SVM 예측
      const analysis = extractFeaturesAndPredict(mono, decodedBuffer.sampleRate);
      console.log('Analysis Result for', filename, ':', analysis);

      updateDashboard(filename, analysis, mono);
    } catch (err) {
      console.error('Audio processing error:', err);
      decisionBadge.className = 'decision-badge';
      badgeIcon.textContent = '❌';
      badgeTitle.textContent = '오디오 분석 실패';
      badgeDesc.textContent = err.message || '오디오 파일을 디코딩하는 중 오류가 발생했습니다.';
      alert(`오디오 분석 중 오류가 발생했습니다: ${err.message}`);
    }
  }

  async function loadSampleWav(sampleUrl, sampleName) {
    try {
      const res = await fetch(sampleUrl);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} - 샘플 파일을 찾을 수 없습니다 (${sampleUrl})`);
      }
      const arrayBuffer = await res.arrayBuffer();
      await processAudioArrayBuffer(arrayBuffer, sampleName);
    } catch (err) {
      console.error('Failed to load sample:', err);
      if (window.location.protocol === 'file:') {
        alert(`[로컬 직접 실행 안내]\n브라우저 보안 정책(CORS)으로 인해 file:// 환경에서는 원클릭 샘플 자동 fetch가 차단될 수 있습니다.\n\n해결 방법: 위의 '파일 업로드' 영역을 클릭하거나 드래그 앤 드롭으로 './samples/' 폴더 안의 WAV 파일을 직접 선택하시면 완벽하게 분석 및 재생됩니다!`);
      } else {
        alert(`샘플 로드 실패: ${err.message}`);
      }
    }
  }

  // ==========================================
  // 10. 이벤트 리스너 등록
  // ==========================================
  // 파일 선택
  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const arrayBuffer = await file.arrayBuffer();
    await processAudioArrayBuffer(arrayBuffer, file.name);
    fileInput.value = ''; // 초기화
  });

  // 드래그 앤 드롭
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const arrayBuffer = await file.arrayBuffer();
    await processAudioArrayBuffer(arrayBuffer, file.name);
  });

  // 원클릭 샘플 테스트 버튼
  btnSampleFootstep.addEventListener('click', () => {
    loadSampleWav('./samples/1-51147-A-25.wav', '1-51147-A-25.wav (발소리 샘플)');
  });

  btnSampleVoice.addEventListener('click', () => {
    loadSampleWav('./samples/03-01-01-01-01-01-01.wav', '03-01-01-01-01-01-01.wav (말소리 샘플)');
  });

  // 오디오 재생 버튼
  btnPlayOrig.addEventListener('click', () => {
    if (isPlayingType === 'orig') {
      stopAudioPlayback();
    } else {
      playAudio('orig');
    }
  });

  btnPlayAnc.addEventListener('click', () => {
    if (isPlayingType === 'anc') {
      stopAudioPlayback();
    } else {
      playAudio('anc');
    }
  });

  btnStop.addEventListener('click', stopAudioPlayback);

  // 초기화 실행
  window.addEventListener('DOMContentLoaded', () => {
    loadSVMModel();
  });
})();
