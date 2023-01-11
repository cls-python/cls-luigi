# Variant 4

This variant implements the pipeline shown in pipeline_to_implement.svg.

In this variant, the resulting pipeline modeling should be closer to the result of my master thesis
to the extent that the selection of the config is tied to the selection of the scoring method
(or vice versa, depending on what is more comprehensible to model).

## Implementation

The structure here is like we have it in variant_1 but this time the concrete config implementations
require the fitting scoring method to be executed. This would result in failing pipelines
where scoring method does not fit the requirements from the selected config.
To prevent the execution of these pipelines, a unique abstract task filter is used to filter them out,
thus not running them in the first place.

## Problems

- with a approach like the one in this variant, it is hard to have AbstractConfigPackages that include other AbstractConfigPackages, because the requirement is defined in the AbstractClass.
- maybe there is an easy solution for that, but i have not yet come up with anything.