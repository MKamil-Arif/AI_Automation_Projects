"""
Topic 6 - Fingerprint Enhancement & Minutiae Matching.

Enhancement (Hong, Wan & Jain 1998 style)
  1. Normalise brightness / contrast, find the finger area (block standard deviation)
  2. Orientation field: smoothed structure tensor of the Sobel gradients (doubled angles)
  3. Gabor filter tuned to the local orientation and ridge wavelength -> clean ridges
  4. Binarise, thin to 1-pixel skeleton
Minutiae
  5. Crossing number on the skeleton: CN = 1 -> ridge ending, CN = 3 -> bifurcation
  6. Drop minutiae near the finger border and pairs that are too close (broken ridges / spurs)
Matching
  7. Every pair (minutia in A, minutia in B, same type) votes for a rotation + translation
     (generalised Hough transform); the best alignments are checked by counting minutiae that
     land within 10 px with a direction within 15 deg -> score = matched^2 / (nA * nB)
     (synthetic samples: same finger >= 0.57, different <= 0.09; real FVC2002 prints score lower:
     same finger 0.21 on average, different fingers 0.01)

Default threshold 9 %: just above every impostor score of FVC2002 DB2_B (no false matches there).
Tested on FVC2002 DB1_B: accuracy / FNMR / FMR and EER in EVALUATION.md.
Run alone:  python -m topics.t06_fingerprint_matching samples/fingerprint_A1.png
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

ENDING, BIFURCATION = 1, 3


@dataclass
class FingerprintAnalysis:
    image: np.ndarray            # input gray image (analysed size)
    mask: np.ndarray             # finger area
    enhanced: np.ndarray         # Gabor output
    skeleton: np.ndarray         # bool
    minutiae: np.ndarray         # N x 4 : x, y, angle (radians, mod pi), type (1 / 3)


@register
class FingerprintMatcher(DIPModule):
    topic_no = 6
    name = "Fingerprint Minutiae Matching"
    description = ("Enhances a fingerprint with orientation-tuned Gabor filters, extracts ridge endings "
                   "and bifurcations (crossing number), and matches two prints by aligning minutiae.")
    samples = ("fingerprint_A1.png", "fingerprint_A2.png", "fingerprint_B1.png", "fingerprint_B2.png")

    WORK_SIZE = 500
    ORIENTATION_BINS = 16

    def parameters(self) -> list[Param]:
        return [
            Param("mode", "Mode", "select", "Enhance & minutiae",
                  options=("Enhance & minutiae", "Match two fingerprints")),
            Param("second", "Second fingerprint (to compare with)", "image", None,
                  visible_when={"mode": "Match two fingerprints"}),
            Param("wavelength", "Ridge spacing (px)", "slider", 9, 5, 16, 1,
                  help="Distance between two neighbouring ridges at the analysed size."),
            Param("threshold", "Match threshold (score %)", "slider", 9, 1, 80, 1,
                  visible_when={"mode": "Match two fingerprints"}),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, second, wavelength, threshold) -> ModuleResult:
        a = self.analyse(image, wavelength, show=True)
        endings, bifurcations = (int((a.minutiae[:, 3] == t).sum()) for t in (ENDING, BIFURCATION))
        if mode == "Enhance & minutiae":
            out = self._draw_minutiae(a)
            return ModuleResult(out, info={"Minutiae": len(a.minutiae), "Ridge endings": endings,
                                           "Bifurcations": bifurcations,
                                           "Finger area": f"{(a.mask > 0).mean():.0%}"})

        if second is None:
            raise ValueError("Upload the second fingerprint to compare with.")
        b = self.analyse(second, wavelength, show=False)
        score, pairs, (angle, shift) = self.match(a.minutiae, b.minutiae)
        output = self._draw_match(a, b, pairs)
        self._add_step("Second print: minutiae", self._draw_minutiae(b))
        return ModuleResult(output, info={
            "Verdict": "MATCH" if score * 100 >= threshold else "NO MATCH",
            "Score": f"{score:.0%}",
            "Matched minutiae": len(pairs),
            "Minutiae A / B": f"{len(a.minutiae)} / {len(b.minutiae)}",
            "Alignment": f"rotate {np.degrees(angle):.0f} deg, shift ({shift[0]:.0f}, {shift[1]:.0f}) px",
        })

    # ----- enhancement ----------------------------------------------------------
    def analyse(self, image: np.ndarray, wavelength: int, show: bool) -> FingerprintAnalysis:
        step = self._add_step if show else (lambda *_: None)
        gray, _ = utils.resize_max(utils.ensure_gray(image), self.WORK_SIZE)
        step("Original", gray)
        norm = ((gray.astype(np.float32) - gray.mean()) / (gray.std() + 1e-6)).astype(np.float32)

        mask = self.segment(norm, wavelength)
        step("1. Finger area (block standard deviation)", mask)

        theta = self.orientation(norm)
        step("2. Orientation field", self._draw_orientation(gray, theta, mask))

        enhanced = self.gabor_enhance(norm, theta, wavelength)
        step("3. Gabor-enhanced ridges", utils.normalize_to_uint8(-enhanced * (mask > 0)))

        ridges = (enhanced < 0) & (mask > 0)                       # ridges are dark -> negative response
        ridges = remove_small_objects(ridges, max_size=wavelength * 3)
        skeleton = skeletonize(ridges)
        step("4. Thinned ridges (skeleton)", (skeleton * 255).astype(np.uint8))

        minutiae = self.extract_minutiae(skeleton, theta, mask, wavelength)
        return FingerprintAnalysis(gray, mask, enhanced, skeleton, minutiae)

    @staticmethod
    def segment(norm: np.ndarray, wavelength: int) -> np.ndarray:
        block = 2 * wavelength
        mean = cv2.blur(norm, (block, block))
        std = np.sqrt(np.maximum(cv2.blur(norm * norm, (block, block)) - mean * mean, 0))
        mask = (std > 0.35).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block | 1, block | 1))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            mask = ((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
        return mask

    @staticmethod
    def orientation(norm: np.ndarray, sigma: float = 6.0) -> np.ndarray:
        """Dominant GRADIENT direction (radians, 0..pi); ridges run perpendicular to it."""
        gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        # doubled-angle representation: averaging 2*theta avoids the 0 / 180 degree jump
        c = cv2.GaussianBlur(gxx - gyy, (0, 0), sigma)
        s = cv2.GaussianBlur(2 * gxy, (0, 0), sigma)
        return (0.5 * np.arctan2(s, c)) % np.pi

    def gabor_enhance(self, norm: np.ndarray, theta: np.ndarray, wavelength: int) -> np.ndarray:
        bins = self.ORIENTATION_BINS
        size = int(wavelength * 2.5) | 1
        idx = np.round(theta / np.pi * bins).astype(int) % bins
        out = np.zeros_like(norm)
        for k in range(bins):
            kernel = cv2.getGaborKernel((size, size), wavelength * 0.45, k * np.pi / bins, wavelength, 1.0, 0)
            kernel -= kernel.mean()                                 # no response to flat areas
            sel = idx == k
            if sel.any():
                out[sel] = cv2.filter2D(norm, cv2.CV_32F, kernel)[sel]
        return out

    # ----- minutiae -------------------------------------------------------------
    def extract_minutiae(self, skeleton: np.ndarray, theta: np.ndarray, mask: np.ndarray,
                         wavelength: int) -> np.ndarray:
        sk = np.pad(skeleton.astype(np.int8), 1)
        # 8 neighbours in circular order P1..P8
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        nb = [np.roll(np.roll(sk, -dy, 0), -dx, 1)[1:-1, 1:-1] for dy, dx in offsets]
        cn = sum(np.abs(nb[i] - nb[(i + 1) % 8]) for i in range(8)) // 2
        inner = cv2.erode(mask, np.ones((3 * wavelength, 3 * wavelength), np.uint8)) > 0   # away from the border
        points = []
        for t in (ENDING, BIFURCATION):
            ys, xs = np.nonzero(skeleton & (cn == t) & inner)
            points += [(x, y, (theta[y, x] + np.pi / 2) % np.pi, t) for x, y in zip(xs, ys)]
        pts = np.array(points, np.float64).reshape(-1, 4)
        if len(pts) < 2:
            return pts
        # minutiae closer than one ridge spacing are mostly noise (broken ridge, spur) -> drop both
        d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return pts[d.min(axis=1) > wavelength]

    # ----- matching -------------------------------------------------------------
    @staticmethod
    def _angle_diff(a, b):
        """Smallest difference between two orientations (mod pi)."""
        d = np.abs(a - b) % np.pi
        return np.minimum(d, np.pi - d)

    def match(self, A: np.ndarray, B: np.ndarray, max_rot: float = np.radians(35),
              dist_tol: float = 10, angle_tol: float = np.radians(15)):
        if len(A) < 3 or len(B) < 3:
            return 0.0, [], (0.0, np.zeros(2))
        # 1. every same-type pair votes for (rotation, translation)
        ia, ib = np.nonzero(A[:, 3][:, None] == B[:, 3][None, :])
        votes = []
        for flip in (0, np.pi):                                     # orientation is only known mod pi
            rot = (B[ib, 2] - A[ia, 2] + flip + np.pi / 2) % np.pi - np.pi / 2
            ok = np.abs(rot) <= max_rot
            c, s = np.cos(rot[ok]), np.sin(rot[ok])
            ax, ay = A[ia[ok], 0], A[ia[ok], 1]
            tx = B[ib[ok], 0] - (c * ax - s * ay)
            ty = B[ib[ok], 1] - (s * ax + c * ay)
            votes.append(np.column_stack([rot[ok], tx, ty]))
        votes = np.vstack(votes)
        if len(votes) == 0:
            return 0.0, [], (0.0, np.zeros(2))
        cells = np.column_stack([np.round(votes[:, 0] / np.radians(5)), np.round(votes[:, 1] / 10),
                                 np.round(votes[:, 2] / 10)]).astype(int)
        uniq, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)

        # 2. check the 10 most voted alignments, keep the one that pairs up the most minutiae
        best = (0, [], (0.0, np.zeros(2)))
        for cell in np.argsort(counts)[::-1][:10]:
            rot, tx, ty = votes[inverse.ravel() == cell].mean(axis=0)
            c, s = np.cos(rot), np.sin(rot)
            moved = np.column_stack([c * A[:, 0] - s * A[:, 1] + tx, s * A[:, 0] + c * A[:, 1] + ty])
            d = np.hypot(moved[:, None, 0] - B[None, :, 0], moved[:, None, 1] - B[None, :, 1])
            close = (d < dist_tol) & (self._angle_diff((A[:, 2] + rot)[:, None], B[None, :, 2]) < angle_tol)
            d = np.where(close, d, np.inf)
            pairs, used_a, used_b = [], set(), set()
            for k in np.argsort(d, axis=None):                     # greedy one-to-one pairing
                i, j = divmod(int(k), d.shape[1])
                if not np.isfinite(d[i, j]):
                    break
                if i not in used_a and j not in used_b:
                    used_a.add(i), used_b.add(j), pairs.append((i, j))
            if len(pairs) > best[0]:
                best = (len(pairs), pairs, (rot, np.array([tx, ty])))
        # matched^2 / (nA * nB): punishes a match that only covers a small part of either print
        score = best[0] ** 2 / (len(A) * len(B))
        return score, best[1], best[2]

    # ----- drawing ----------------------------------------------------------------
    def _draw_orientation(self, gray, theta, mask, step: int = 14) -> np.ndarray:
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for y in range(step // 2, gray.shape[0], step):
            for x in range(step // 2, gray.shape[1], step):
                if mask[y, x]:
                    a = theta[y, x] + np.pi / 2                     # draw the ridge direction
                    dx, dy = 6 * np.cos(a), 6 * np.sin(a)
                    cv2.line(out, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (0, 0, 255), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _draw_minutiae(a: FingerprintAnalysis) -> np.ndarray:
        out = cv2.cvtColor(a.image, cv2.COLOR_GRAY2BGR)
        out[a.skeleton] = (0.5 * out[a.skeleton] + (0, 90, 0)).astype(np.uint8)
        for x, y, ang, t in a.minutiae:
            colour = (0, 0, 255) if t == ENDING else (255, 120, 0)
            cv2.circle(out, (int(x), int(y)), 5, colour, 1, cv2.LINE_AA)
            cv2.line(out, (int(x), int(y)), (int(x + 10 * np.cos(ang)), int(y + 10 * np.sin(ang))), colour, 1, cv2.LINE_AA)
        utils.put_label(out, f"endings (red) {int((a.minutiae[:, 3] == ENDING).sum())}  "
                             f"bifurcations (blue) {int((a.minutiae[:, 3] == BIFURCATION).sum())}",
                        (6, 18), (255, 255, 255), 0.45)
        return out

    def _draw_match(self, a: FingerprintAnalysis, b: FingerprintAnalysis, pairs) -> np.ndarray:
        left, right = self._draw_minutiae(a), self._draw_minutiae(b)
        h = max(left.shape[0], right.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out = np.hstack([pad(left), pad(right)])
        off = left.shape[1]
        for i, j in pairs:
            p = (int(a.minutiae[i, 0]), int(a.minutiae[i, 1]))
            q = (int(b.minutiae[j, 0]) + off, int(b.minutiae[j, 1]))
            cv2.line(out, p, q, (0, 200, 0), 1, cv2.LINE_AA)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(FingerprintMatcher)
