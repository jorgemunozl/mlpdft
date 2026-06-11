#set page(paper: "a4", flipped: true, margin: 0.5in)

#align(center)[
  #text(2em, weight: "bold")[FITSNAP METRICS] \
  #v(1em)
]

#align(center, table(
  columns: 9,
  stroke: 0.5pt,
  inset: 5pt,

  // === HEADER ROW 1: Group headers ===
  table.cell(fill: luma(230), rowspan: 2, [*Metric*]),
  table.cell(fill: luma(230), colspan: 2, [*LIF64*]),
  table.cell(fill: luma(230), colspan: 3, [*LIFINTERFACE*]),
  table.cell(fill: luma(230), colspan: 3, [*LIWITHF*]),

  // === HEADER ROW 2: Individual dataset names ===
  table.cell(fill: luma(210), [ISOLATED]),
  table.cell(fill: luma(210), [KJPAW V2]),
  table.cell(fill: luma(210), [KJPAW V1]),
  table.cell(fill: luma(210), [KJPAW NPT]),
  table.cell(fill: luma(210), [KJPAW NPT V2]),
  table.cell(fill: luma(210), [V3]),
  table.cell(fill: luma(210), [ISOLATED]),
  table.cell(fill: luma(210), [NPT FINAL]),

  // === ENERGY SECTION ===
  table.cell(fill: luma(220), colspan: 9, [*Energy (eV)*]),

  [MAE],
  [$1317.8194$],
  [$2908.4603$],
  [$10723.6836$],
  [$17886.8874$],
  [$6418.7791$],
  [$6342.0009$],
  [$10308.8928$],
  [$6238.8993$],

  [RMSE],
  [$1361.5060$],
  [$2909.6136$],
  [$10759.3499$],
  [$19038.5364$],
  [$6741.2427$],
  [$6342.0042$],
  [$10308.9282$],
  [$6239.2208$],

  [MaxAE],
  [$1704.0213$],
  [$3106.9338$],
  [$12609.2213$],
  [$29208.5294$],
  [$12391.3947$],
  [$6353.8602$],
  [$10359.3796$],
  [$6344.1368$],

  // === FORCES SECTION ===
  table.cell(fill: luma(220), colspan: 9, [*Forces (eV/Å)*]),

  [MAE (all)],
  [$0.2067$],
  [$0.1370$],
  [$0.4607$],
  [$1.2388$],
  [$0.5334$],
  [$0.0544$],
  [$0.0934$],
  [$0.1373$],

  [RMSE (all)],
  [$0.2899$],
  [$0.1774$],
  [$0.6799$],
  [$1.8607$],
  [$0.7430$],
  [$0.0722$],
  [$0.1109$],
  [$0.1970$],

  [MAE (x)],
  [$0.2041$],
  [$0.1417$],
  [$0.4665$],
  [$1.1965$],
  [$0.5208$],
  [$0.0526$],
  [$0.0938$],
  [$0.1287$],

  [MAE (y)],
  [$0.2058$],
  [$0.1362$],
  [$0.4945$],
  [$1.1936$],
  [$0.5265$],
  [$0.0570$],
  [$0.0903$],
  [$0.1337$],

  [MAE (z)],
  [$0.2102$],
  [$0.1330$],
  [$0.4210$],
  [$1.3263$],
  [$0.5528$],
  [$0.0536$],
  [$0.0961$],
  [$0.1495$],

  [MaxAE],
  [$0.9478$],
  [$0.8019$],
  [$4.9474$],
  [$20.2048$],
  [$5.5680$],
  [$0.2810$],
  [$0.2638$],
  [$3.6398$],
))
