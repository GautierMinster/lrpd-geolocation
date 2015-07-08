# encoding=UTF-8

# stdlib
import logging
import math
import random

# 3p
import cv2
import numpy as np

# project
import util.image


log = logging.getLogger(__name__)


class RMSMatcher(object):
    """Match RMSMaps to a database of reference RMSMaps.

    Given a test image, for example a printed tourist map, represented by its
    road segments as a RMSMap, match it to a pre-computed database of RMSMaps,
    to determine where the map is from, ie determine:
      - the city
      - the area within the city
      - the scale of the map
      - the rotation of the map

    Matching is done using FLANN (each reference map is a training descriptor
    set), and RANSAC is used for robust matching and transform estimation.
    """

    def __init__(self, rmsm_refs={}):
        """Initializes the matcher.

        Args:
            rmsm_refs: a dictionary, with reference map names as keys, and the
                corresponding RMSMaps as values. It is possible to add reference
                maps after object creation as well, so this can be empty or
                incomplete.
        """
        self._init_matcher()
        self.refs = []
        self.names = []
        self.indices = {}

        self.add_refs(rmsm_refs)

    def add_refs(self, rmsm_refs):
        """Add reference maps to the training data set.

        This method may be called any number of times to progressively add
        reference maps.

        Args:
            rmsm_refs: a dictionary with reference map names as keys and the
                corresponding RMSMaps as values
        """
        for name, rmsm in rmsm_refs.iteritems():
            self.indices[name] = len(self.refs)
            self.names.append(name)
            self.refs.append(rmsm)
            self.flann.add([np.asarray(rmsm.descs, np.float32)])

        # Only train if we actually added descriptors (necessary, Flann WILL
        # crash if we train on the empty set)
        if rmsm_refs:
            self.flann.train()

    def match(self, msm_test, snn_threshold=0.95):
        """Matches a test RMSMap to the reference maps.

        Args:
            rmsm_test: a RMSMap representation of the input map
            snn_threshold: threshold used to discard undistinctive matches,
                where the best match is close to the second best match (snn,
                second nearest neigbor), according to the following condition:
                dist_first >= dist_second * snn_threshold

        Returns:
            A dictionary of the form {ref_name: [matches for this ref map]}.
            Each list of matches has been filtered using the above threshold,
            and a match is a pair (kp_ref, kp_test) of Keypoint objects.
        """
        if not self.refs:
            log.error('No reference maps provided as training data.')
            raise Exception('Nothing to match the test map to.')

        # Get the 2 nearest neighbors for each input descriptor
        k = 2
        matches = self.flann.knnMatch(
            np.asarray(msm_test.descs, np.float32),
            k=k
        )

        good_matches = {name: [] for name in self.indices.iterkeys()}

        for [f, s] in matches:
            if f.distance < s.distance * snn_threshold:
                kp_ref = self.refs[f.imgIdx].get_desc_keypoint(
                    f.trainIdx
                )
                kp_test = msm_test.get_desc_keypoint(
                    f.queryIdx
                )
                good_matches[self.names[f.imgIdx]].append(
                    (kp_ref, kp_test)
                )

        return good_matches

    def estimate_model(self, test_map_size, matches,
            dtol=1.0, atol=0.1, maxits=1000, auto_accept_inlier_count=30):
        """Estimate the model, ie the similarity between test and reference maps

        The model to be estimated is a a similarity transformation (plus a
        translation). There are thus 4 degrees of freedom: 2 for translation,
        1 for scaling, and 1 for rotation. Pairs of matches are used to estimate
        the model using RANSAC.

        The transform is defined as:
        p_ref = s * R(theta) * p_test + t
            where:
              - p_ref and p_test are 2D points in the ref and test maps
              - s is a scaling factor
              - R(theta) is a rotation matrix, of angle theta
              - t is a 2D translation (in ref pixel units)

        Attrs:
            test_map_size: a (height, width) couple, representing the dimensions
                of the test map being matched (used for conditioning the data)
            matches: a dictionary as returned by the match method
            dtol: the tolerance on the distance when looking for inliers, as a
                proportion of the keypoint radius
            atol: the tolerance on the angle difference when looking for
                inliers, as a proportion of the angle range [-pi; pi[
            maxits: the maximum number of iterations to perform
            auto_accept_inlier_count: the number of inliers above which a model
                is automatically accepted and the random selection process is
                considered done. This effectively avoids continuing the random
                iterations when a great inlier set has already been found.

        Returns:
            A couple (inliers, models), where inliers and models are both dicts
            with the reference map names as keys, and as values, numpy 4-vectors
            of the form [tx, ty, sr, o], where:
              - (tx, ty): the translation
              - sr: the scaling (scale ratio of ref over test)
              - o: the rotation angle (ref - test, ie angle 0 in the test map
                means angle o in the reference)
        """
        # Setup
        #######
        # Some reference maps may not have been matched at all in the nearest
        # neigbor search, so we'll process those which have

        # List of reference map indices
        refs = []
        # List of raw matches for each ref
        matches_ref = []
        # List of models estimated for each ref
        models_ref = []
        # List of inliers for each ref
        inliers_ref = []
        # Total number of raw matches, across all refs
        matches_count = 0
        # Probability vector for weighted random choice of ref
        p = []
        # Conditioning factors for each ref
        c_refs = []

        # Initialize the above variables
        for name in matches.iterkeys():
            matches_in_ref = len(matches[name])
            # If we don't have enough data to even fit a model, don't bother
            # using this map
            if matches_in_ref < 2:
                continue
            refidx = self.indices[name]
            refs.append(refidx)
            matches_ref.append(matches[name])
            models_ref.append(None)
            inliers_ref.append([])
            matches_count += matches_in_ref
            p.append(float(matches_in_ref))
            c_refs.append(float(max(
                self.refs[refidx].dmap_origin.img.shape
            )))
        # Normalize the p vector to actual probabilities
        p = map(lambda x: x/float(matches_count), p)

        # Conditioning factor for the test map
        c_test = float(max(test_map_size))

        # Store if a good enough match was found (its index is stored)
        #   When this happens, the random iterations will be stopped
        match_found = None

        # Random iterations
        ###################

        it = 0
        # If no matches at all, abandon
        if len(refs) == 0:
            return None, None

        while it < maxits:
            it += 1
            # Choose a random reference map
            # The random choice is weighted by the number of raw matches for
            # each reference
            refidx = np.random.choice(len(refs), p=p)
            m = matches_ref[refidx]
            c_ref = c_refs[refidx]

            elems = random.sample(xrange(len(m)), 2)
            model = self._ransac_fit_model_lstsq(m, elems, c_ref, c_test)
            if model is None:
                continue

            # Iterate over all the matches to find inliers
            inliers = self._ransac_find_inliers(
                m, model, c_ref, c_test, dtol, atol
            )

            if len(inliers) > len(inliers_ref[refidx]):
                models_ref[refidx] = model
                inliers_ref[refidx] = inliers
                # If we reached a good enough model, make note of it and stop
                # the random iterations
                if len(inliers) >= auto_accept_inlier_count:
                    match_found = refidx
                    break

        # Final model fitting
        #####################
        # For reference maps where this is relevant (ie only one if enough
        # matches were found that we are certain, or multiple otherwise), we'll
        # fit a model using least squares on the inliers, and iterate until the
        # process converges (ie no more adding or removing inliers).

        models_final = {}
        inliers_final = {}

        for refidx, ref in enumerate(refs):
            # If we found a good match, don't perform least squares fitting for
            # other reference maps
            if match_found is not None and match_found != refidx:
                continue

            if len(inliers_ref[refidx]) == 0:
                continue

            m = matches_ref[refidx]
            c_ref = c_refs[refidx]

            # Limit the number of least squares fittings we perform
            max_lstsq_its = 10
            lstsq_it = 0
            # Store whether there was a change of the set of inliers
            inliers_changed = True
            while inliers_changed and lstsq_it < max_lstsq_its:
                lstsq_it += 1

                model = self._ransac_fit_model_lstsq(m, inliers_ref[refidx],
                    c_ref, c_test)

                # If we couldn't estimate a model, skip
                if model is None:
                    inliers_ref[refidx] = []
                    break

                new_inliers = self._ransac_find_inliers(m, model, c_ref, c_test,
                    dtol, atol)

                # If we end up without inliers, abandon
                if len(new_inliers) == 0:
                    inliers_ref[refidx] = []
                    break

                # See if the set of inliers changed
                #   The two lists of inliers are sorted ascending, so make use
                #   of that when comparing
                inliers_changed = False
                # If the length is different, easy
                if len(inliers_ref[refidx]) != len(new_inliers):
                    inliers_changed = True
                else:
                    for k1, k2 in zip(inliers_ref[refidx], new_inliers):
                        if k1 != k2:
                            inliers_changed = True
                            break

                inliers_ref[refidx] = new_inliers
                models_ref[refidx] = model

            if len(inliers_ref[refidx]) == 0:
                continue

            models_final[self.names[ref]] = models_ref[refidx]
            inliers_final[self.names[ref]] = [m[k] for k in inliers_ref[refidx]]

        return inliers_final, models_final

    def _ransac_fit_model_lstsq(self, m, matches, c_ref, c_test):
        """Computes using least-squares a model, based on a list of matches.

        The system to solve is of the form a * (u, v, tx, ty) = b, where b is a
        column vector of all the reference keypoints, and a is the vertical
        concatenation of the matrices:
            |kp_test.x  -kp_test.y  1  0|
            |kp_test.y   kp_test.x  0  1|
        u and v define the scaling + rotation:
                  / u = s * cos(theta)
                  \ v = s * sin(theta)

        Args:
            m: the entire list of matches
            matches: a list of match indices to use to fit the model; must be of
                size at least 2
            c_ref: the conditioning factor for the reference
            c_test: the conditioning factor for the test

        Returns:
            A numpy 4-vector, of the form [tx, ty, s, o], as specified in the
            documentation of the estimate_model() method, or None if the matches
            don't yield a model.
        """
        count = len(matches)

        if count < 2:
            return None

        if count == 2:
            # If we're in the random iteration loop and trying to fit a model
            # using two matches, check that we actually can use these two
            if (m[matches[0]][0].p == m[matches[1]][0].p).all() \
                    or (m[matches[0]][1].p == m[matches[1]][1].p).all():
                return None

        a = np.empty((2*count, 4), dtype=np.float_)
        b = np.empty(2*count, dtype=np.float_)
        # We'll make sure the maximum weight is 1, to preserve the effects of
        # the conditioning
        max_weight = 0.
        for row, k in enumerate(matches):
            mkr = m[k][0]
            mkt = m[k][1]
            # Define the weight for this match
            # Use an F1-score of the inverse scales of the two keypoints
            wr = c_ref / mkr.s
            wt = c_test / mkt.s
            w = 2.*wr*wt/(wr+wt)
            max_weight = max(max_weight, w)
            a[2*row:2*row+2, :] = w * np.asarray(
                [[mkt.p[0]/c_test, -mkt.p[1]/c_test, 1., 0.],
                 [mkt.p[1]/c_test,  mkt.p[0]/c_test, 0., 1.]],
                dtype=np.float_)
            b[2*row:2*row+2] = w * np.asarray(
                [mkr.p[0]/c_ref, mkr.p[1]/c_ref], dtype=np.float_
            )

        # Adjust the weights
        a /= max_weight
        b /= max_weight

        # Compute the conditioned fitted model
        fitted_model = np.linalg.lstsq(a, b)[0]
        # Compute our format of model (tx, ty, s, o):
        model = np.empty(4, dtype=np.float_)
        # Translation
        model[:2] = fitted_model[2:4]
        # Scaling
        model[2] = np.linalg.norm(fitted_model[:2])
        # Angle
        model[3] = np.arctan2(fitted_model[1], fitted_model[0])
        if model[3] >= math.pi:
            model[3] -= 2.*math.pi

        # Reverse the conditioning
        model[:2] *= c_ref
        model[2] *= c_ref / c_test

        return model

    def _ransac_find_inliers(self, matches, model, c_ref, c_test, dtol, atol):
        """Iterates over all matches to find model inliers.

        Args:
            matches: a list of matches, ie (keypoint_ref, keypoint_test) pairs
            model: a numpy 4-vector representing the transformation model
            c_ref: the conditioning factor for the reference map
            c_test: the conditioning factor for the test map
            dtol: distance tolerance for inlier detection
            atol: angle tolerance for inlier detection

        Returns:
            An array of indices of the matches list, corresponding to the
            inliers of the model.
        """
        # The rotation matrix
        rot = np.array([[math.cos(model[3]), -math.sin(model[3])],
                        [math.sin(model[3]),  math.cos(model[3])]],
                        dtype=np.float_)

        inliers = []
        for k, mk in enumerate(matches):
            # First, see if the orientation matches. This should eliminate the
            # bulk of the outliers
            # Difference between estimated orientation and real reference one
            #   Be careful that even though the model's rotation is in
            #   [-pi; pi[, the orientation of segments is only determined up to
            #   a modulo pi.
            #   So when adding the model's orientation to the test orientation,
            #   make it modulo pi, and make sure the difference is lower than
            #   pi/2
            delta_o = mk[0].o - ((mk[1].o + model[3]) % math.pi)
            while delta_o < -math.pi/2.:
                delta_o += math.pi
            while delta_o >= math.pi/2:
                delta_o -= math.pi
            # Error in the orientation, in percentage
            err_o = abs(delta_o / (2.*math.pi))

            if err_o > atol:
                continue

            # If the angle fits, then check the position

            # Image of the test keypoint by the assumed model
            p_est_ref = model[2] * rot.dot(mk[1].p) + model[:2]
            # Image of the reference keypoint by the assumed model
            #   The inverse rotation is simply the transpose
            p_est_test = rot.T.dot(mk[0].p - model[:2]) / model[2]

            # Error in the position in the reference, in pixel units
            #   Condition it, since we'll combine the two errors
            err_p_ref = np.linalg.norm(p_est_ref-mk[0].p) / c_ref
            # Same, in the test map
            err_p_test = np.linalg.norm(p_est_test-mk[1].p) / c_test

            # Conditioned scales
            s_ref  = mk[0].s / c_ref
            s_test = mk[1].s / c_test

            if err_p_ref**2. + err_p_test**2. < \
                    dtol**2. * (s_ref**2. + s_test**2.):
                inliers.append(k)

        return inliers

    def identify(self, mm_ref, test_map_size, inliers, models,
            min_metric_size=200., max_metric_size=10.0e3,
            min_inliers=3., first_to_others_threshold=0.6):
        """Decide which reference map (if any) the test map is from.

        Args:
            mm_ref: a dictionary of reference metric maps
            test_map_size: a (height, width) couple, the size of the test
                region
            inliers: a dictionary, containing for each reference map name
                (the keys), a list of inlier matches
            models: a dictionary, containing for each reference map name the
                estimated model
            min_metric_size: the minimum metric size of a valid match (ie a set
                of inliers which shows the metric size of the test map to be
                lower than this is discarded)
            max_metric_size: similarly, metric size above which a match will be
                discarded

        Returns:
            A dictionary of reference map names, where:
              - an empty dict means there is no match
              - a dict with a single element means we're pretty sure about our
                choice
              - a dict with multiple elements means we're not too sure between
                all of those, but they are the most probable, and the value
                of each key is a confidence index in [0,1], such that the sum
                of all confidences is 1.
        """
        # Maximum dimension of the test map
        test_size = float(max(test_map_size))

        # Only consider reference maps for which a model was fitted
        candidates = [self.indices[name] for name in models.iterkeys()]

        # Check the min and max metric sizes
        filtered_candidates = []
        for ref in candidates:
            name = self.names[ref]
            rmsm_ref = self.refs[ref]
            test_metric_size = mm_ref[name].get_metric_distance(
                test_size * models[name][2]
            )
            if test_metric_size < min_metric_size \
                    or test_metric_size > max_metric_size:
                continue
            filtered_candidates.append(ref)
        candidates = filtered_candidates

        # Compute a weighted number of inliers, where the weighting is based
        # on the proportion of the test descriptor region actually inside the
        # test region, and the proportion of the reference descriptor actually
        # inside the test region
        weighted_inlier_count = {ref: 0. for ref in candidates}
        for ref in candidates:
            name = self.names[ref]
            ref_map_size = self.refs[ref].dmap_origin.img.shape
            for kp_ref, kp_test in inliers[name]:
                # top left point, intersected with image boundaries
                tl_test = [
                    max(0,kp_test.p[0]-kp_test.s+1),
                    max(0,kp_test.p[1]-kp_test.s+1)
                ]
                tl_ref = [
                    max(0,kp_ref.p[0]-kp_ref.s+1),
                    max(0,kp_ref.p[1]-kp_ref.s+1),
                ]
                # bottom right point
                br_test = [
                    min(test_map_size[1]-1, kp_test.p[0]+kp_test.s),
                    min(test_map_size[0]-1, kp_test.p[1]+kp_test.s)
                ]
                br_ref = [
                    min(ref_map_size[1]-1, kp_ref.p[0]+kp_ref.s),
                    min(ref_map_size[0]-1, kp_ref.p[1]+kp_ref.s)
                ]
                # The contribution of this match is the area inside the test
                # image, divided by the total theoretical area of the descriptor
                w_test = (br_test[0]-tl_test[0])*(br_test[1]-tl_test[1]) \
                    / (2.*kp_test.s)**2.
                w_ref = (br_ref[0]-tl_ref[0])*(br_ref[1]-tl_ref[1]) \
                    / (2.*kp_ref.s)**2.
                # Compute the F1-score of the 2 weights
                weighted_inlier_count[ref] += 2.*w_test*w_ref / (w_test+w_ref)

        # Discard candidates with less than min_inliers inliers
        filtered_candidates = []
        for ref in candidates:
            if weighted_inlier_count[ref] >= min_inliers:
                filtered_candidates.append(ref)
            else:
                weighted_inlier_count[ref] = 0.
        candidates = filtered_candidates

        if not candidates:
            return {}

        # Return all candidates within first_to_others_threshold
        first = max(weighted_inlier_count.itervalues())
        filtered_candidates = []
        for ref in candidates:
            if weighted_inlier_count[ref] >= first_to_others_threshold * first:
                filtered_candidates.append(ref)
            else:
                weighted_inlier_count[ref] = 0.
        candidates = filtered_candidates

        # Return the final results, a list of possibilities, with an associated
        # confidence (in [0,1])
        total = sum(weighted_inlier_count.itervalues())
        result = {
            self.names[ref]: weighted_inlier_count[ref]/total
            for ref in candidates
        }

        return result

    def draw_matches(self, refname, rmsm_test, matches, model=None,
            img_test=None, groundtruth=None, lines=False, circles=False,
            sw=2, lw=3, cw=2, locw=5, gtw=5, sc=(150,150,150), lc=(0,0,255),
            cc=(0,0,200), locc=(0,0,255), gtc=(255,0,0), bgc=(255,255,255)):
        """Draw a set of matches between a reference image and a test image.

        Args:
            refname: the name of the reference RMSMap to use
            rmsm_test: the test RMSMap
            matches: the set of matches to display (as either the array of
                matches for this reference map, or the dictionary returned by
                match() directly)
            model: if provided, the estimated
            img_test: if provided, a color-image (ie 3 channels BGR) of the test
                data. When it's provided, the segments for the test map are not
                drawn
            groundtruth: a numpy 4-vector, of the same format as the model
            lines: if True, draw lines between the matches
            circles: if True, draw the descriptor radius
            sw, lw, cw: segment, match line, and match circle widths
            sc, lc, cc: segment, match line, and match circle colors
            locw, gtw: location and groundtruth boxes widths
            locc, gtc: location and groundtruth boxes colors
            bgc: background color

        Returns:
            A 3-channel image showing the matches between the reference map and
            the test map.
        """
        if type(matches) is dict:
            matches = matches[refname]
        if type(model) is dict:
            model = model[refname]

        out = self._draw_maps(
            self.refs[self.indices[refname]], rmsm_test, box=True, sw=sw, sc=sc,
            bc=lc, bgc=bgc
        )
        h_ref, w_ref = self.refs[self.indices[refname]].dmap_origin.img.shape
        h_test, w_test = rmsm_test.dmap_origin.img.shape

        # Draw the image if we have one
        if img_test is not None:
            out[:img_test.shape[0], w_ref:] = img_test

        # Draw the matches
        for (kpref, kptest) in matches:
            if lines:
                cv2.line(out,
                    tuple(map(int, kpref.p)),
                    (int(kptest.p[0])+w_ref,int(kptest.p[1])),
                    lc, lw
                )
            if circles:
                cv2.circle(out,
                    tuple(map(int, kpref.p)), int(kpref.s), cc, cw
                )
                cv2.circle(out,
                    (int(kptest.p[0])+w_ref,int(kptest.p[1])), int(kptest.s),
                    cc, cw
                )

        # Draw the estimated location and the groundtruth location if provided
        model_list = [m for m in [model, groundtruth] if m is not None]
        width_list = [locw, gtw]
        color_list = [locc, gtc]
        # The 4 corners of the test image, as columns, in the following order:
        #   top left, top right, bottom left, bottom right
        # Use homogeneous coordinates for convenience
        corners_test = np.array(
            [[0, w_test-1,    0    , w_test-1],
             [0,    0    , h_test-1, h_test-1],
             [1,    1    ,    1    ,    1    ]], dtype=np.float_
        )
        for m, w, c in zip(model_list, width_list, color_list):
            # Compute the transform
            transform = np.array(
                [[m[2]*math.cos(m[3]), -m[2]*math.sin(m[3]), m[0]],
                 [m[2]*math.sin(m[3]),  m[2]*math.cos(m[3]), m[1]]],
                dtype=np.float_)
            # Warp, and transform to ints
            cw = np.asarray(transform.dot(corners_test), dtype=np.int_)

            # top
            cv2.line(out, (cw[0,0],cw[1,0]), (cw[0,1],cw[1,1]), c, w)
            # left
            cv2.line(out, (cw[0,0],cw[1,0]), (cw[0,2],cw[1,2]), c, w)
            # right
            cv2.line(out, (cw[0,1],cw[1,1]), (cw[0,3],cw[1,3]), c, w)
            # bottom
            cv2.line(out, (cw[0,2],cw[1,2]), (cw[0,3],cw[1,3]), c, w)

        return out

    def _draw_maps(self, rmsm1, rmsm2, box=True, sw=1, bw=2, sc=(150,150,150),
            bc=(0,0,255), bgc=(255,255,255)):
        """Draws two RMSMaps side-by-side.

        Args:
            rmsm1: the left RMSMap
            rmsm2: the right RMSMap
            box: if True, a bounding box will be drawn around the right map
            sw: segment line width
            bw: box line width
            sc: segment color
            bc: box color
            bgc: background color

        Returns:
            A 3-channel image with both images drawn side-by-side.
        """
        dmap1 = rmsm1.dmap_origin
        dmap2 = rmsm2.dmap_origin

        h_out = max(dmap1.img.shape[0], dmap2.img.shape[0])
        w_out = dmap1.img.shape[1] + dmap2.img.shape[1]
        w1 = dmap1.img.shape[1]
        h2 = dmap2.img.shape[0]
        out = np.empty([h_out, w_out, 3], dtype=np.uint8)
        out[:,:] = bgc
        util.image.draw_segments(dmap1.roads, out, width=sw, color=sc)
        util.image.draw_segments(dmap2.roads, out[:,w1:], width=sw, color=sc)

        if not box:
            return out

        # Top side
        cv2.line(out, (w1, 0), (w1, h2-1), bc, bw)
        cv2.line(out, (w1, 0), (out.shape[1]-1, 0), bc, bw)
        cv2.line(out, (out.shape[1]-1, h2-1), (out.shape[1]-1, 0),
            bc, bw)
        cv2.line(out, (out.shape[1]-1, h2-1), (w1, h2-1),
            bc, bw)

        return out

    def overlay_ref_on_image(self, refname, img, model, lw=2, color=(0,0,255)):
        """Draw a reference map onto an image of a matched map.

        Args:
            refname: the name of the reference map to draw
            img: an image (1 or 3 channels) where the reference segments will
                be drawn (not in place, on a 3-channel copy)
            model: the transformation model, as returned by the model estimation
                (can be the raw dictionary that was returned)
            lw: width in pixels of the segments to draw
            color: color to use when drawing the segments

        Returns:
            A new 3-channel image, copied from img, where the segments from the
            reference map refname were drawn.
        """
        shape = img.shape
        if len(shape) == 2:
            out = cv2.merge([img, img, img])
        elif len(shape) == 3 and shape[2] == 1:
            tmp = img.reshape(shape[:2])
            out = cv2.merge([tmp, tmp, tmp])
        elif len(shape) == 3 and shape[2] == 3:
            out = img.copy()
        else:
            raise Exception("Don't know what to do with this matrix shape: {}"\
                .format(shape)
            )

        if type(model) == dict:
            model = model[refname]

        ref = self.refs[self.indices[refname]]

        # Compute the coordinate transform to go from reference map coordinates
        # to image coordinates
        segs = ref.dmap_origin.roads.lines.copy()
        # Reverse the modelled transform
        tx, ty = model[0], model[1]
        segs -= np.array([tx, ty, tx, ty], dtype=np.float_)
        # Scale
        segs /= model[2]
        # Rotate in the other direction
        rot = np.array([[ math.cos(model[3]), math.sin(model[3])],
                        [-math.sin(model[3]), math.cos(model[3])]],
                       dtype=np.float_)
        # Split the segs in start and stop points
        start, stop = segs[:, :2].T, segs[:, 2:4].T
        segs = np.concatenate((rot.dot(start).T, rot.dot(stop).T), axis=1)
        # Round the coordinates, and convert to integer
        segs = np.asarray(np.around(segs), dtype=np.int_)

        for s in segs:
            cv2.line(out, (s[0],s[1]), (s[2],s[3]), color, lw)

        return out

    def _init_matcher(self):
        """Instantiate and configure the FLANN matcher."""
        FLANN_INDEX_KDTREE = 1
        index_params = {
            FLANN_INDEX_KDTREE: {
                'algorithm': FLANN_INDEX_KDTREE,
                'trees': 30
            }
        }
        search_params = {
            'checks': 3000
        }

        self.flann = cv2.FlannBasedMatcher(
            index_params[FLANN_INDEX_KDTREE],
            search_params
        )

