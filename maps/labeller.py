# encoding=UTF-8

# stdlib
import copy

class FrozenLabeller(Exception):
    pass

class MultiSetLabeller(object):
    """Manage multi-sets of elements of an element space.

    Instances of this are used to assign labels (unsigned integers) to subsets
    of an element space (unsigned integers too). The single difference between
    this and a union-find (multi-set) data structure is that when two sets are
    joined, they produce a third one which is the union of the two, instead of
    merging into a single set.

    After a phase of construction of the labelling, we want to be able to obtain
    efficiently the elements of the subset corresponding to a label. Since we
    are going to iterate on those elements and process them anyway, instead of
    returning some sort of container for these elements, we'll just return an
    iterable that yields them.
    Note that this is a multi-set, so each element will be returned with a
    corresponding counter.

    WARNING: it is irrelevant to our purpose that there is a unicity of
    label <=> subset. Which means that depending on how the elements were added,
    there may be several labels that correspond to the same subset of elements.

    The construction phase is carried out by iteratively extending existing
    labels using multi-subsets of elements.

    We represent the data as a tree:
    - the root is the empty set, label 0
    - each node has a label, and thus corresponds to a multi-set of elements. It
      also has :
        - the label of its parent node
        - a list of pairs of elements that are part of the multiset (first
          element), and each of those elements' total count (second element).
          This means that when going up the hierarchy, there is no need to keep
          a counter for each element: as soon as we encounter an element, the
          associated value is the final one
        - the total number of elements (multi-elements counted multiple times)
          in the multi-set
        - a dictionary of children nodes , identified by some sort of user-
          provided ID (keys), and the corresponding labels as values (optional,
          but makes the structure a lot more space efficient when working from
          an existing labelling of multi-sets, because the same extension will
          produce the same label). When the MultiSetLabeller is frozen (ie it's
          put in read-only mode), the dictionaryies of children are destroyed,
          they are no longer needed. The tree nodes then become 3-tuples instead
          of 4-tuples.
    """

    def __init__(self):
        """Create the MultiSetLabeller."""
        # Node 0 (the empty set) has:
        #  - no parent
        #  - no added elements
        #  - 0 elements
        #  - no children for now
        self.labels = [ (-1, [], 0, {}) ]
        self.label_count = 1
        self.frozen = False

    def add(self, elems, l=0, uid=None):
        """Compute a label for the addition of elems to label l.

        Args:
            elems: the elements to add to the label, as a list of (e, c) pairs,
                where e is an element, and c is the count to add. If multiple
                pairs correspond to the same element e, the counts will be added
            l: the label to add the elements to. By default, the empty set
            uid: (optional) a unique identifier (typically, a label from another
                labelling), used if provided to avoid create a new label for a
                set that already exists and is based on l.

        Returns:
            A new label that corresponds to the union of elems and l.
        """
        lp, ladded, lcount, lchildren = self.labels[l]

        # Return the existing label for this join if it exists
        if uid is not None:
            try:
                return lchildren[uid]
            except KeyError:
                pass

        # Create the stub for the added field of the new label
        added = {}
        for e, c in elems:
            new_c = added.get(e, 0) + c
            added[e] = new_c

        count = sum(added.itervalues()) + lcount

        if count == 0:
            return l

        # We are going to go up the tree and count the occurences of each
        # element e of elems in the label l
        # We'll decrement notseen as we update the values in previous. Previous
        # should only be updated once per element, so notseen will reach 0 when
        # we're all done
        notseen = len(added)
        previous = {e: None for e in added}
        lcurr = l
        while notseen > 0 and lcurr != 0:
            # Get the data and update lcurr
            lcurr, lcurr_added, _, _ = self.labels[lcurr]
            for e, c in lcurr_added:
                if e in previous and not previous[e]:
                    previous[e] = c
                    notseen -= 1

        # If we reached the root, set the remaining unfound values to 0
        if lcurr == 0:
            for e, c in previous.iteritems():
                if not previous[e]:
                    previous[e] = 0

        # Merge the counts and make a list
        added = [(e, c+previous[e]) for e, c in added.iteritems()]
        lnew = self.label_count
        self.label_count += 1
        if uid is not None:
            lchildren[uid] = lnew
        self.labels.append( (l, added, count, {}) )

        return lnew

    def get(self, l):
        """Returns a (total_count, elements generator) pair.

        The returned generator allows iteration over pairs of (element, count).

        Args:
            l: the label to return

        Returns:
            A (total_count, generator) couple, where total_count is the total
            number of elements (multi-elements counted multiple times), and
            generator yields (element, count) pairs.
        """
        data = self.labels[l]
        parent = data[0]
        total = data[2]
        # If the label we're processing has label 0 (the empty set) for parent,
        # (which happens most of the time in our setup), then we can simply
        # yield all of the elements without worrying
        if parent == 0:
            return total, self._elements_at_node(l)
        else:
            return total, self._recursive_elements_generator(l)

    def _elements_at_node(self, l):
        """A generator that yields all the elements of a specific node.

        Unlike _recursive_elements_generator, this method does not go up the
        tree. It simply yields all of the elements stored at node l.
        In the case of a label l whose parent is 0, this is exactly the same,
        only less hassle.

        Args:
            l: the node whose elements to yield

        Yields:
            Pairs of the form (element, count), where count is the number of
            occurrences of element in the multi-set.
        """
        for pair in self.labels[l][1]:
            yield pair

    def _recursive_elements_generator(self, l):
        """A generator that yields the elements of a label.

        This generator is dubbed recursive because it goes up the tree, while
        keeping track of the elements it has already seen, yielding the new ones
        as it finds them.

        Args:
            l: the label whose element we'll return

        Yields:
            Pairs of the form (element, count), where count is the number of
            occurrences of element in the multi-set.
        """
        seen = set()
        while l != 0:
            l, elems = self.labels[l][0:2]
            for pair in elems:
                if not pair[0] in seen:
                    seen.add(pair[0])
                    yield pair

    def freeze(self):
        """Freezes the labeller, ie makes it read-only.

        Subsequent calls to add will yield undefined results, so don't.
        """
        if self.frozen:
            return

        for i in xrange(self.label_count):
            parent, added, count, children = self.labels[i]
            # Make sure the parent is of type int, this could be False when
            # interacting with numpy
            self.labels[i] = int(parent), added, count

        self.frozen = True

    def to_dict(self):
        """Returns a dictionary representation suitable for JSON formatting."""
        return {
            'frozen': self.frozen,
            'labels': self.labels,
            'label_count': self.label_count
        }

    @classmethod
    def from_dict(cls, d):
        """Create a MultiSetLabeller instance from a dict representation."""
        msl = MultiSetLabeller()
        msl.frozen = d['frozen']
        msl.labels = d['labels']
        msl.label_count = d['label_count']
        return msl

