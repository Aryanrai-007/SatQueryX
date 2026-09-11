export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export async function fileToDataUrl(file: File) {
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Unable to read file"));
    reader.readAsDataURL(file);
  });
}

export async function imageDataUrl(file: File) {
  if (file.type.startsWith("image/") && !file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
    return fileToDataUrl(file);
  }
  return null;
}

export async function compareImages(a: File, b: File) {
  const [aUrl, bUrl] = await Promise.all([imageDataUrl(a), imageDataUrl(b)]);
  if (!aUrl || !bUrl) return null;

  const [ia, ib] = await Promise.all([
    loadImage(aUrl),
    loadImage(bUrl),
  ]);
  const width = 256;
  const height = 256;
  const ca = document.createElement("canvas");
  const cb = document.createElement("canvas");
  ca.width = cb.width = width;
  ca.height = cb.height = height;
  const xa = ca.getContext("2d");
  const xb = cb.getContext("2d");
  if (!xa || !xb) return null;
  xa.drawImage(ia, 0, 0, width, height);
  xb.drawImage(ib, 0, 0, width, height);
  const pa = xa.getImageData(0, 0, width, height).data;
  const pb = xb.getImageData(0, 0, width, height).data;
  let changed = 0;
  let sum = 0;
  for (let i = 0; i < pa.length; i += 4) {
    const d = (Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1]) + Math.abs(pa[i + 2] - pb[i + 2])) / (255 * 3);
    sum += d;
    if (d > 0.12) changed += 1;
  }
  return {
    changedFraction: changed / (width * height),
    meanDifference: sum / (width * height),
  };
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Unable to decode image"));
    img.src = src;
  });
}
