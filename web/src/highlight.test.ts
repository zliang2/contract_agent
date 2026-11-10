import { segments } from "./highlight";

const join = (s: { text: string }[]) => s.map((x) => x.text).join("");

test("pieces concatenate back to the text and mark the exact span", () => {
  const text = "The laws of Delaware govern this Agreement.";
  const start = text.indexOf("Delaware");
  const segs = segments(text, [{ start, end: start + 8, active: true }]);
  expect(join(segs)).toBe(text);
  expect(segs.filter((s) => s.kind === "active").map((s) => s.text)).toEqual(["Delaware"]);
});

test("active wins over cited where spans overlap; out-of-range spans are clipped", () => {
  const text = "abcdefghij";
  const segs = segments(text, [
    { start: 2, end: 6, active: false },
    { start: 4, end: 8, active: true },
    { start: 9, end: 50, active: false },
  ]);
  expect(join(segs)).toBe(text);
  expect(segs.map((s) => [s.text, s.kind])).toEqual([
    ["ab", "plain"],
    ["cd", "cited"],
    ["efgh", "active"],
    ["i", "plain"],
    ["j", "cited"],
  ]);
});
