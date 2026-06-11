#set page(margin: 1in)

#align(center, table(
  columns: 6,
  stroke: 0.5pt,
  inset: 6pt,

  // === HEADER ROW 1 ===
  table.cell(fill: luma(230), rowspan: 2, [*Metric*]),
  table.cell(fill: luma(230), colspan: 3, [*Dataset*]),
  table.cell(fill: luma(230), rowspan: 2, [*LIF KJPAW*]),
  table.cell(fill: luma(230), rowspan: 2, [*LIWITHF V3*]),

  // === HEADER ROW 2 ===
  table.cell(fill: luma(230), [LIF64 ISOLATED]),
  table.cell(fill: luma(230), [LIF64 KJPAW V2]),
  table.cell(fill: luma(230), [LIFINTERFACE KJPAW V1]),

  // === ENERGY SECTION ===
  table.cell(fill: luma(210), colspan: 6, [*Energy (eV)*]),

  [MAE],
  [$1317.8194$],
  [$2908.4603$],
  [$10723.6836$],
  [$6342.0009$],
  [$6342.0009$],

  [RMSE],
  [$1361.5060$],
  [$2909.6136$],
  [$10759.3499$],
  [$6342.0042$],
  [$6342.0042$],

  [MaxAE],
  [$1704.0213$],
  [$3106.9338$],
  [$12609.2213$],
  [$6353.8602$],
  [$6353.8602$],

  // === FORCES SECTION ===
  table.cell(fill: luma(210), colspan: 6, [*Forces (eV/Å)*]),

  [MAE (all)],
  [$0.2067$],
  [$0.1370$],
  [$0.4607$],
  [$0.0544$],
  [$0.0544$],

  [RMSE (all)],
  [$0.2899$],
  [$0.1774$],
  [$0.6799$],
  [$0.0722$],
  [$0.0722$],

  [MAE (x)],
  [$0.2041$],
  [$0.1417$],
  [$0.4665$],
  [$0.0526$],
  [$0.0526$],

  [MAE (y)],
  [$0.2058$],
  [$0.1362$],
  [$0.4945$],
  [$0.0570$],
  [$0.0570$],

  [MAE (z)],
  [$0.2102$],
  [$0.1330$],
  [$0.4210$],
  [$0.0536$],
  [$0.0536$],

  [MaxAE],
  [$0.9478$],
  [$0.8019$],
  [$4.9474$],
  [$0.2810$],
  [$0.2810$],
))
