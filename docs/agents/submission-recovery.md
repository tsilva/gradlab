# Lost admission acknowledgement

Never rerun a launch whose acknowledgement is missing. A client error does not
prove that no Run was admitted. Inspect immutable evidence before recovery:

```sh
gradlab experiment status --submission-key STABLE_KEY \
  --goal-slug ENVIRONMENT/GOAL --created-since UTC_TIMESTAMP --json
```

The timestamp must precede the launch, include an explicit UTC offset and not
be in the future. This read-only command searches the selected goal's current
manifest-event inventory, then reads root manifests. Current admission writes
the event before the manifest. A recent event without its manifest is unresolved,
not proof of absence. Invalid events and inventory limits fail closed. A
`not_admitted` result applies only to the declared goal and time interval.

If a matching Run exists, inspect it by Run ID and use supported `resume-submit`
or retry recovery only after the required coordinator and lease checks. Multiple
matches or unresolved candidates require review. Do not create another Run.

For a failed sequential batch, a reviewed `not_admitted` lookup can reconcile its
unacknowledged intent. Preserve the complete lookup JSON in the job work directory
as `INTENT_KEY-reviewed-absence.json`, adding `reviewed: true` only after inspecting
it. The result must have an empty match and unresolved inventory, cover at least
the job's creation time, and have been observed after the failed child command
exited. The failed command receipt must still exist. Generic job retry does not
erase uncertainty or authorize resubmission by itself.

Start only the scoped `experiment-batch` flusher after review. It archives the
failed command receipt and records the proof hash before returning the intent
to pending admission. An admitted Run retains its original identity, source,
worker lifetime and bounds. An access outage never permits a credential fallback
or automatic approval of changed access settings.
