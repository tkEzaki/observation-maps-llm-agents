# Stage B 追加診断レポート

実施日: 2026-07-23  
対象: 初回run 4,800応答 + 独立再現run 4,800応答  
追加API費用: USD 0

## 1. 結論

既存9,600応答を用いた追加解析は、以下を支持した。

1. 集中度による第一調波の消失は、固定pairwise kernelのvon Mises
   convolutionでは説明できない。
2. 第二調波の増幅は、同じ固定kernel scalingと定量的に整合する。
3. したがって、現時点の最も具体的な仮説は、狭い分布で第一調波channelだけが
   選択的に抑制され、第二調波channelが保持される
   `concentration-induced harmonic switching` である。
4. 低次構造は強く再現するが、run間差は有限samplingだけよりわずかに大きい。
5. jaggednessには観測grid由来の高周波成分が含まれるが、単純なbin center/edge
   parityだけでは説明できない。
6. 狭い場の第一調波消失は、全般的な`stay`化ではない。整列点付近では停止する
   一方、それ以外では高いactivityを保ったままadvance/retardの配置が
   第二調波型へ組み替わっている。

この結果は、representation invariance実験を次の有料実験とする判断を強める。
大規模collective simulationへ直行する段階ではない。

## 2. 固定pairwise kernelの検定

固定kernel \(H(\phi)\) の線形重ね合わせなら、von Mises場に対する第\(m\)調波は

\[
a_m(\kappa)
=c_m\frac{I_m(\kappa)}{I_0(\kappa)}
\]

に従う。数値積分で得た\(\kappa=2\)から12へのscaleは、

- \(m=1\): 1.37205;
- \(m=2\): 2.78083.

であった。

### 第一調波

- broad pooled \(a_1(2)=0.57144\);
- 固定kernel予測 \(a_1(12)=0.78405\);
- narrow pooled実測 \(a_1(12)=-0.03852\);
- 実測−予測: -0.82257;
- bootstrap 95%区間: [-0.87795, -0.76773].

ゼロ差は区間に含まれず、固定kernel予測は明確に棄却された。

### 第二調波

- broad pooled \(a_2(2)=0.11397\);
- 固定kernel予測 \(a_2(12)=0.31693\);
- narrow pooled実測 \(a_2(12)=0.33176\);
- 実測−予測: 0.01483;
- bootstrap 95%区間: [-0.08576, 0.11230].

第二調波は固定kernel scalingと整合した。

この対照は、全応答が一様に変化したのではなく、第一調波が選択的に抑制された
ことを示す。表現変更後にも同じ対照が維持されれば、論文の中心結果になり得る。

## 3. 有限samplingによるrun間差の検定

各offsetについて

\[
Q=\sum_\delta
\frac{[\hat g_1(\delta)-\hat g_2(\delta)]^2}
{\widehat{\mathrm{Var}}_1(\delta)/50+
 \widehat{\mathrm{Var}}_2(\delta)/50}
\]

を計算した。さらに各offsetのpooled三値確率を共通の真値と仮定し、二つの
50標本runを10,000回再生成した。

### Broad field

- 観測 \(Q=64.66\);
- null平均 \(Q=47.12\);
- parametric-bootstrap \(p=0.0479\);
- 観測curve correlation: 0.9489;
- 同一応答則からのcorrelation 95%範囲: [0.9395, 0.9758];
- 観測相関のnull分布内percentile: 12.3%.

相関は同一応答則から十分起こり得る。Qは境界的で、一部offsetの差が通常の
samplingよりやや大きい可能性がある。

### Narrow field

- 観測 \(Q=71.05\);
- null平均 \(Q=48.05\);
- parametric-bootstrap \(p=0.0204\);
- 観測curve correlation: 0.8992;
- 同一応答則からのcorrelation 95%範囲: [0.9050, 0.9613];
- 観測相関のnull分布内percentile: 1.11%.

narrow run間差はsampling-only nullよりわずかに大きい。低次調波と強応答の符号
は再現しているため中心結果は維持されるが、seed block、実行時刻、backendの
非決定性による小さなoverdispersionを今後の不確実性に含める必要がある。

したがって「sampling ceilingに完全到達」とは主張せず、

> 低次構造と応答方向は再現するが、pointwise振幅にはmultinomial samplingを
> 超える小さなrun間変動がある

と記述するのが適切である。

## 4. Offset parityとfull Fourier spectrum

48点応答に対しNyquist \(m=24\)までのDFTを計算した。

### Broad field

- parity係数 \(\langle g_q(-1)^q\rangle=0.0683\);
- bootstrap 95%区間: [0.0460, 0.0906];
- \(m=24\) amplitude: 0.0683;
- 一点差の平均絶対値: 0.3467;
- 二点差の平均絶対値: 0.4338;
- grid一段あたりに換算した二点差: 0.2169.

parityは統計的に検出されるが、第一調波amplitude 0.574よりかなり小さい。
主要な高周波成分には\(m=10,23,11,14,12\)なども含まれ、単純な二点周期だけ
ではない。

### Narrow field

- parity係数: 0.0142;
- bootstrap 95%区間: [-0.0083, 0.0373];
- \(m=24\) amplitude: 0.0142;
- 一点差の平均絶対値: 0.4267;
- 二点差の平均絶対値: 0.3796;
- grid一段あたりに換算した二点差: 0.1898.

narrow fieldでは単純な\(m=24\) parityは検出されなかった。一方、
\(m=23\) amplitudeは0.242で、主成分\(m=2\)の0.348に次いで大きい。

結論として、

- broadの\(m=1\)とnarrowの\(m=2\)は安定した低次構造;
- \(m=23\)周辺を含む高周波成分はobservation-grid fingerprint候補;
- center/edge parity単独ではjaggednessを説明できない

と分離するのが妥当である。

## 5. 三値確率とactivity

\[
A(\delta)=p_{\rm advance}(\delta)+p_{\rm retard}(\delta)
=1-p_{\rm stay}(\delta)
\]

をpooledデータから計算した。

### Broad field

- 平均activity: 0.910;
- activity範囲: 0から1;
- \(\delta=0\)でのactivity: 0.020;
- \(\delta=\pi\)でのactivity: 0.240.

### Narrow field

- 平均activity: 0.848;
- activity範囲: 0から1;
- \(\delta=0\)でのactivity: 0;
- \(\delta=\pi\)でのactivity: 0.110.

両条件とも整列点ではほぼdeterministic `stay`となる。しかし全offset平均では
activityが高い。narrow fieldで第一調波が消えた原因は一様なactivity低下では
なく、activeなadvance/retard応答が第二調波型に再配置されたことにある。

`g=0`は常に同じ状態ではない。今後の解析では各offsetの
\(p_+,p_0,p_-\)とactivityを保存し、平均actionだけへ縮約しない。

## 6. Even componentの解釈

pooled constant termは、

- broad \(a_0=0.0179\);
- narrow \(a_0=0.0121\)

と小さい。一方、even-component RMSは約0.25と大きい。

これはoffset-independentなadvance biasではなく、configuration-dependentな
非反対称応答を示す。ただし現在はsynthetic many-body fieldへの一体応答であり、
pairwise nonreciprocityの証拠ではない。二体mirror条件で
\(g(\delta)+g(-\delta)\)を直接測定する必要がある。

## 7. 次の有料実験のprimary endpoints

Representation sensitivity testではpointwise correlationを主判定にしない。
primary endpointsは、

\[
a_1(\kappa=2),\quad a_1(\kappa=12),\quad
a_2(\kappa=2),\quad a_2(\kappa=12)
\]

とし、中心仮説

\[
a_1(2)>0,\qquad a_1(12)\simeq0,\qquad
a_2(12)>a_2(2)>0
\]

が以下の表現を超えて維持されるか検証する。

1. half-bin shifted discretization;
2. 12-binと48-bin;
3. decimal precision変更;
4. integer percentageまたは固定総和count;
5. low-order circular moments.

同時に\(m=23,24\)をgrid-fingerprint endpointとして記録する。低次switchingが
残り、高周波だけが変化することが最も強い結果である。

## 8. Small-Nに進む前の条件

small-\(N\) histogramはStage Bの滑らかな場と異なる。fractionsに正規化しても
値の刻みから\(N\)を推測できるため、以下をStage C前に固定する。

- representative sparse histogramsへの直接応答測定;
- smoothingまたはpseudocountの有無と固定値;
- collective入力とStage B stimulus manifoldの距離;
- \(r_1,r_2,r_3,\Omega_{\rm coll},\tau_{\rm social},p_{\rm stay}\)の保存.

## 9. 並列化の確認

過去二つのAPI runのthroughputは、

- 初回: 約24.8 calls/s;
- 再現: 約26.1 calls/s

であった。4,800件を約3分で完了しており、直列実行ではない。

CPU使用率が低いのは、Stage BがCPU simulationではなくnetwork I/O待ちの
thread並列だからである。各workerの大部分はAPI応答待ちとなり、20並列でも
CPUを占有しない。

runnerには今後、各callのelapsed time、合計task time、wall time、
effective parallelism、calls/sを保存する計測を追加した。ローカルslow-backend
検証では、

- configured concurrency: 4;
- 同時active call: 3以上;
- effective parallelism: 2.60

を確認した。最初の1件を直列preflightにすることと、残り7件が二waveになる
小規模テストを考えると妥当である。

CPU負荷の高いoffline collective simulationsはまだ実行していない。Stage C
実装時には、独立runをprocess poolで並列化し、run内agent更新をvectorizeする。
LLM collective dynamicsでは時刻ごとの同期barrierを保ちつつ、同一snapshotの
agent callだけを並列化する。

## 10. 生成物

- `analysis/stage_b_extended/extended_diagnostics.json`
- `analysis/stage_b_extended/full_spectrum.csv`
- `analysis/stage_b_extended/action_probabilities.csv`
- `analysis/stage_b_extended_diagnostics.py`
- 本文書 `docs/STAGE_B_EXTENDED_DIAGNOSTICS.md`

10,000回のparametric/bootstrap simulationを使用した。追加API呼び出しと
追加API費用はない。
