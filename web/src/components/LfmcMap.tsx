import { useState, useMemo } from "react";
import { DeckGL } from "@deck.gl/react";
import { ColumnLayer, GeoJsonLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/maplibre";
import { AmbientLight, DirectionalLight, LightingEffect } from "@deck.gl/core";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Site, ColorMode } from "../types";
import { lfmcColor, uncColor, VEG_COLOR, VEG_LABEL } from "../lib/color";

const DARK = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json";
const INITIAL = { longitude: 136, latitude: -29.5, zoom: 4.0, pitch: 50, bearing: -11 };

const lighting = new LightingEffect({
  ambient: new AmbientLight({ color: [255, 255, 255], intensity: 1.1 }),
  sun: new DirectionalLight({ color: [255, 255, 255], intensity: 1.6, direction: [-1, -3, -1] }),
});

function colorFor(d: Site, mode: ColorMode): [number, number, number] {
  if (mode === "lfmc") return lfmcColor(d.lfmc);
  if (mode === "unc") return uncColor(d.unc);
  return VEG_COLOR[d.veg];
}

export default function LfmcMap({ sites, outline }: { sites: Site[]; outline: unknown | null }) {
  const [mode, setMode] = useState<ColorMode>("lfmc");
  const [hover, setHover] = useState<{ x: number; y: number; o: Site } | null>(null);

  const layers = useMemo(() => {
    const ls: any[] = [];
    if (outline)
      ls.push(
        new GeoJsonLayer({
          id: "australia",
          data: outline as never,
          stroked: true,
          filled: true,
          getFillColor: [34, 25, 18, 235],
          getLineColor: [150, 110, 70, 200],
          getLineWidth: 1,
          lineWidthUnits: "pixels",
        })
      );
    ls.push(
      new ColumnLayer<Site>({
        id: "sites",
        data: sites,
        diskResolution: 16,
        radius: 15000,
        extruded: true,
        pickable: true,
        elevationScale: 4200,
        radiusUnits: "meters",
        getPosition: (d) => [d.lon, d.lat],
        getElevation: (d) => d.unc + 2,
        getFillColor: (d) => [...colorFor(d, mode), 235] as [number, number, number, number],
        material: { ambient: 0.55, diffuse: 0.6, shininess: 32, specularColor: [60, 64, 70] },
        onHover: (info) =>
          setHover(info.object ? { x: info.x, y: info.y, o: info.object as Site } : null),
        updateTriggers: { getFillColor: mode },
        transitions: { getFillColor: 300, getElevation: 600 },
      })
    );
    return ls;
  }, [sites, mode, outline]);

  return (
    <div className="hero">
      <DeckGL initialViewState={INITIAL} controller={true} layers={layers} effects={[lighting]}>
        <Map mapStyle={DARK} attributionControl={false} />
      </DeckGL>

      <div className="hero-overlay">
        <div className="kicker">Globe-LFMC 2.0 field sites · Sentinel-2 (Digital Earth Australia)</div>
        <h1>Live fuel moisture in Australian forests</h1>
        <p>
          Each column is a field-measured LFMC site. Colour shows the moisture estimate;
          height shows the model's uncertainty at that site. Drag to rotate, scroll to zoom.
        </p>
      </div>

      <div className="panel controls">
        <div className="title">Colour by</div>
        <div className="seg">
          {(["lfmc", "unc", "veg"] as ColorMode[]).map((m) => (
            <button key={m} className={mode === m ? "active" : ""} onClick={() => setMode(m)}>
              {m === "lfmc" ? "LFMC" : m === "unc" ? "Uncertainty" : "Veg type"}
            </button>
          ))}
        </div>
        <div className="hint">Column height encodes per-site model uncertainty (random-forest tree spread).</div>
      </div>

      <Legend mode={mode} />

      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <div className="name">{hover.o.site}</div>
          <div className="row"><span>Vegetation</span><b>{VEG_LABEL[hover.o.veg]}</b></div>
          <div className="row"><span>Bioregion</span><b>{hover.o.bioregion}</b></div>
          <div className="row"><span>LFMC (mean)</span><b>{hover.o.lfmc}%</b></div>
          <div className="row"><span>Predicted</span><b>{hover.o.pred}%</b></div>
          <div className="row"><span>Uncertainty</span><b>±{hover.o.unc}%</b></div>
          <div className="row"><span>Measurements</span><b>{hover.o.n}</b></div>
        </div>
      )}
    </div>
  );
}

function Legend({ mode }: { mode: ColorMode }) {
  if (mode === "veg") {
    return (
      <div className="panel legend">
        <div className="title">Vegetation type</div>
        {Object.keys(VEG_LABEL).map((v) => (
          <div className="veg-row" key={v}>
            <span className="dot" style={{ background: `rgb(${VEG_COLOR[v].join(",")})` }} />
            {VEG_LABEL[v]}
          </div>
        ))}
      </div>
    );
  }
  const isUnc = mode === "unc";
  const grad = isUnc
    ? "linear-gradient(90deg, rgb(150,116,70), rgb(210,140,60), rgb(206,80,46), rgb(168,40,34))"
    : "linear-gradient(90deg, rgb(92,38,24), rgb(168,72,34), rgb(212,130,48), rgb(236,184,96), rgb(246,224,168))";
  return (
    <div className="panel legend">
      <div className="title">{isUnc ? "Model uncertainty (±%)" : "LFMC (% dry weight)"}</div>
      <div className="bar" style={{ background: grad }} />
      <div className="ticks">
        {(isUnc ? ["2", "15", "30", "50"] : ["40", "90", "140", "220"]).map((t) => <span key={t}>{t}</span>)}
      </div>
    </div>
  );
}
