"""
Topic 18 - Copy-Move Forgery Detection (a part of the image copied and pasted elsewhere).

Block matching (Fridrich et al., 2003)
  1. Slide a BxB window over the gray image (every pixel position)
  2. Feature of each block = its low-frequency DCT coefficients, quantized
     (quantization makes the feature tolerate small JPEG noise)
  3. Skip flat blocks (sky, walls) - they match each other everywhere
  4. Sort the features lexicographically -> identical blocks become neighbours in the list
  5. Neighbours that are far apart in the image give a shift vector (dx, dy)
  6. A real copy produces MANY block pairs with the SAME shift; random look-alike blocks give
     scattered shifts. Shifts with >= min_matches pairs are reported as copy-move.
Keypoints (Amerini et al., 2011 style)
  SIFT keypoints are matched against the image itself (2nd/3rd neighbour ratio test), then
  RANSAC keeps pairs that follow one geometric transform - also catches rotated/scaled copies.

Run alone:  python -m topics.t18_copy_move_forgery samples/copymove_forged.png
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix: coefficients = D @ block @ D.T"""
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    d = np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0] /= np.sqrt(2)
    return d.astype(np.float32)


def zigzag(n: int, count: int) -> np.ndarray:
    """Flat indices of the first `count` coefficients in JPEG zig-zag (low -> high frequency)."""
    cells = sorted(((i, j) for i in range(n) for j in range(n)),
                   key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 else p[0]))
    return np.array([i * n + j for i, j in cells[:count]])


@register
class CopyMoveDetector(DIPModule):
    topic_no = 18
    name = "Copy-Move Forgery Detection"
    description = ("Detects regions that were copied and pasted inside the same image, using "
                   "DCT block matching (or SIFT keypoint matching).")
    samples = ("copymove_forged.png", "copymove_forged_jpeg75.jpg", "copymove_authentic.png")

    # Bigger images are shrunk first (block matching is O(pixels)). Shrinking makes the copy's
    # shift non-integer, so blocks no longer match exactly - keep this as large as practical.
    WORK_SIZE = 800
    N_COEFFS = 16
    COLOUR_A, COLOUR_B = (0, 0, 255), (255, 160, 0)
    BLOCK = {"visible_when": {"method": "Block matching (DCT)"}}
    KEYPT = {"visible_when": {"method": "Keypoints (SIFT)"}}

    def parameters(self) -> list[Param]:
        return [
            Param("method", "Method", "select", "Block matching (DCT)",
                  options=("Block matching (DCT)", "Keypoints (SIFT)")),
            Param("block", "Block size", "select", 8, options=(8, 16), **self.BLOCK),
            Param("quant", "Quantization step (higher = tolerates JPEG)", "slider", 6, 1, 20, 1, **self.BLOCK),
            Param("flat_std", "Skip flat blocks (std below)", "slider", 4.0, 0.0, 20.0, 0.5, **self.BLOCK),
            Param("min_matches", "Min block pairs with the same shift", "slider", 120, 5, 400, 5, **self.BLOCK,
                  help="120: authentic photos reached at most 74 (tuned on 40 TF-Flowers photos)."),
            Param("ratio", "Ratio test (lower = stricter)", "slider", 0.6, 0.3, 0.9, 0.05, **self.KEYPT),
            Param("min_pairs", "Min matched keypoint pairs", "slider", 8, 3, 50, 1, **self.KEYPT),
            Param("min_distance", "Min distance between copies (px)", "slider", 30, 8, 150, 2),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, block, quant, flat_std, min_matches, ratio, min_pairs,
                min_distance) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._add_step("Input (analysed size)", img)

        if method.startswith("Block"):
            mask_a, mask_b, arrows, details = self.block_matching(gray, int(block), quant, flat_std,
                                                                  min_matches, min_distance)
        else:
            mask_a, mask_b, arrows, details = self.keypoint_matching(img, gray, ratio, min_pairs, min_distance)

        forged = bool(arrows)
        output = self._annotate(img, mask_a, mask_b, arrows, forged)
        self._add_step("Detected regions mask", cv2.bitwise_or(mask_a, mask_b))
        info = {
            "Verdict": "FORGED" if forged else "Authentic",
            "Copied area": f"{(cv2.bitwise_or(mask_a, mask_b) > 0).mean():.1%}",
            **details,
        }
        return ModuleResult(output, info=info)

    # ----- method 1: DCT block matching ---------------------------------------
    def block_features(self, gray: np.ndarray, block: int, quant: float):
        """Quantized low-frequency DCT of every overlapping block + its position and std."""
        windows = sliding_window_view(gray.astype(np.float32), (block, block))
        ny, nx = windows.shape[:2]
        d, zz = dct_matrix(block), zigzag(block, self.N_COEFFS)
        feats = np.empty((ny * nx, self.N_COEFFS), np.int32)
        std = np.empty(ny * nx, np.float32)
        rows_per_chunk = max(1, 60000 // nx)                    # chunks keep memory low
        for y0 in range(0, ny, rows_per_chunk):
            blocks = windows[y0:y0 + rows_per_chunk].reshape(-1, block, block)
            s = slice(y0 * nx, y0 * nx + len(blocks))
            feats[s] = np.round((d @ blocks @ d.T).reshape(len(blocks), -1)[:, zz] / quant)
            std[s] = blocks.std(axis=(1, 2))
        ys, xs = np.divmod(np.arange(ny * nx), nx)
        return feats, np.column_stack([xs, ys]), std

    def block_matching(self, gray, block, quant, flat_std, min_matches, min_distance):
        feats, pos, std = self.block_features(gray, block, quant)
        textured = std >= flat_std
        self._add_step("1. Blocks used (white) - flat blocks skipped",
                       self._block_map(gray.shape, pos[textured], block))
        feats, pos = feats[textured], pos[textured]

        order = np.lexsort(feats.T[::-1])                       # lexicographic sort of rows
        feats, pos = feats[order], pos[order]
        pairs_a, pairs_b = [], []
        for off in range(1, 5):                                 # compare with the next 4 rows
            same = np.all(np.abs(feats[off:] - feats[:-off]) <= 1, axis=1)
            idx = np.nonzero(same)[0]
            a, b = pos[idx], pos[idx + off]
            far = np.hypot(*(b - a).T) >= min_distance
            pairs_a.append(a[far])
            pairs_b.append(b[far])
        a, b = np.concatenate(pairs_a), np.concatenate(pairs_b)

        # normalise direction so (A -> B) and (B -> A) count as the same shift
        shift = b - a
        flip = (shift[:, 0] < 0) | ((shift[:, 0] == 0) & (shift[:, 1] < 0))
        a[flip], b[flip] = b[flip], a[flip].copy()
        shift = b - a
        self._add_step("2. All candidate matching blocks", self._block_map(gray.shape, np.vstack([a, b]), block))

        if len(shift):
            vectors, inverse, counts = np.unique(shift, axis=0, return_inverse=True, return_counts=True)
            inverse = inverse.ravel()
        else:
            vectors, inverse, counts = np.zeros((0, 2), int), np.zeros(0, int), np.zeros(0, int)
        accepted = np.nonzero(counts >= min_matches)[0]
        self._add_step("3. Shift-vector histogram (circles = accepted)",
                       self._shift_histogram(gray.shape, vectors, counts, accepted))

        mask_a = np.zeros(gray.shape, np.uint8)
        mask_b = np.zeros(gray.shape, np.uint8)
        arrows = []
        for v in accepted:
            sel = inverse == v
            mask_a[a[sel, 1], a[sel, 0]] = 255
            mask_b[b[sel, 1], b[sel, 0]] = 255
            centre = np.array([block / 2, block / 2])
            arrows.append((a[sel].mean(axis=0) + centre, b[sel].mean(axis=0) + centre))
        kernel = np.ones((block, block), np.uint8)              # top-left corner -> whole block
        mask_a = cv2.dilate(mask_a, kernel, anchor=(block - 1, block - 1))
        mask_b = cv2.dilate(mask_b, kernel, anchor=(block - 1, block - 1))

        top = counts[np.argsort(counts)[::-1][:3]] if len(counts) else []
        details = {
            "Blocks compared": int(len(order)),
            "Candidate pairs": int(len(shift)),
            "Copied shifts": ", ".join(f"({vectors[v][0]}, {vectors[v][1]}): {counts[v]} pairs"
                                       for v in accepted) or "-",
            "Top shift counts": ", ".join(str(c) for c in top) or "-",
        }
        return mask_a, mask_b, arrows, details

    # ----- method 2: SIFT keypoints -------------------------------------------
    def keypoint_matching(self, img, gray, ratio, min_pairs, min_distance):
        sift = cv2.SIFT_create(nfeatures=4000)
        kps, des = sift.detectAndCompute(gray, None)
        self._add_step(f"1. SIFT keypoints ({len(kps)})",
                       cv2.drawKeypoints(img, kps, None, (0, 255, 255)))
        empty = np.zeros(gray.shape, np.uint8)
        if des is None or len(kps) < 10:
            return empty, empty.copy(), [], {"Keypoints": len(kps), "Matched pairs": 0}

        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, des, k=4)
        seen, src, dst = set(), [], []
        for m in knn:
            # drop the point itself (an exact copy has distance 0 too, so self is not always 1st)
            others = [x for x in m if x.trainIdx != x.queryIdx]
            if len(others) < 2:
                continue
            m1, m2 = others[0], others[1]
            pair = tuple(sorted((m1.queryIdx, m1.trainIdx)))
            if m1.distance < ratio * m2.distance and pair not in seen:
                p, q = np.array(kps[m1.queryIdx].pt), np.array(kps[m1.trainIdx].pt)
                if np.hypot(*(q - p)) >= min_distance:
                    seen.add(pair)
                    if q[0] < p[0] or (q[0] == p[0] and q[1] < p[1]):
                        p, q = q, p                             # same direction rule as blocks
                    src.append(p)
                    dst.append(q)

        inliers = np.zeros(len(src), bool)
        if len(src) >= 3:
            _, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst),
                                                 method=cv2.RANSAC, ransacReprojThreshold=4)
            if inl is not None:
                inliers = inl.ravel().astype(bool)
        src, dst = np.array(src).reshape(-1, 2)[inliers], np.array(dst).reshape(-1, 2)[inliers]

        lines = img.copy()
        for p, q in zip(src, dst):
            cv2.line(lines, tuple(map(int, p)), tuple(map(int, q)), (0, 255, 0), 1, cv2.LINE_AA)
        self._add_step(f"2. RANSAC-consistent pairs ({len(src)})", lines)

        mask_a, mask_b, arrows = empty, empty.copy(), []
        if len(src) >= min_pairs:
            for pts, mask in ((src, mask_a), (dst, mask_b)):
                cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 255)
            mask_a = cv2.dilate(mask_a, np.ones((9, 9), np.uint8))
            mask_b = cv2.dilate(mask_b, np.ones((9, 9), np.uint8))
            arrows = [(src.mean(axis=0), dst.mean(axis=0))]
        return mask_a, mask_b, arrows, {"Keypoints": len(kps), "Matched pairs": int(len(src))}

    # ----- evaluation helper (for the report) ---------------------------------
    @staticmethod
    def pixel_scores(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
        """Pixel-level precision / recall / F1 of a detected mask against a ground-truth mask.
        The predicted mask is resized to the ground truth if the image was shrunk for analysis."""
        if predicted.shape[:2] != ground_truth.shape[:2]:
            predicted = cv2.resize(predicted, ground_truth.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        p, g = predicted > 0, ground_truth > 0
        tp = np.logical_and(p, g).sum()
        precision = tp / max(p.sum(), 1)
        recall = tp / max(g.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    # ----- drawing ------------------------------------------------------------
    @staticmethod
    def _block_map(shape, corners: np.ndarray, block: int) -> np.ndarray:
        mark = np.zeros(shape, np.uint8)
        if len(corners):
            mark[corners[:, 1], corners[:, 0]] = 255
        return cv2.dilate(mark, np.ones((block, block), np.uint8), anchor=(block - 1, block - 1))

    @staticmethod
    def _shift_histogram(shape, vectors, counts, accepted) -> np.ndarray:
        h, w = shape
        canvas = np.zeros((2 * h + 1, w + 1), np.float32)       # dx: 0..w, dy: -h..h
        if len(vectors):
            canvas[vectors[:, 1] + h, vectors[:, 0]] = np.log1p(counts)
        canvas = cv2.dilate(canvas, np.ones((7, 7), np.uint8))   # make single points visible
        vis = cv2.applyColorMap(utils.normalize_to_uint8(canvas), cv2.COLORMAP_INFERNO)
        for v in accepted:
            cv2.circle(vis, (int(vectors[v][0]), int(vectors[v][1] + h)), 12, (0, 255, 0), 2)
        cv2.line(vis, (0, h), (w, h), (90, 90, 90), 1)
        return vis

    def _annotate(self, img, mask_a, mask_b, arrows, forged) -> np.ndarray:
        out = img.copy()
        for mask, colour in ((mask_a, self.COLOUR_A), (mask_b, self.COLOUR_B)):
            sel = mask > 0
            out[sel] = (0.55 * out[sel] + 0.45 * np.array(colour)).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, colour, 2, cv2.LINE_AA)
        for p, q in arrows:
            cv2.arrowedLine(out, tuple(map(int, p)), tuple(map(int, q)), (255, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.05)
        text, colour = ("FORGED: copy-move detected", (80, 80, 255)) if forged else ("No copy-move found", (120, 255, 120))
        utils.put_label(out, text, (10, 26), colour, 0.65)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(CopyMoveDetector)
