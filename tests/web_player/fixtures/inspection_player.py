"""Complete player with deterministic recorded-read, bitmap and peer delivery gates."""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from aiohttp import web

from tests.test_policy_bundle import write_bundle
from tests.web_player.fixtures.selection_player import (
    ROUTE,
    SelectionCatalog,
    SelectionHost,
    SelectionLoader,
    SelectionPlayer,
)


class InspectionPlayer(SelectionPlayer):
    async def page(self, request):
        response = await super().page(request)
        return web.Response(
            text=response.text.replace(
                '<script type="module" src="/assets/app.js">',
                '<script type="module" src="/assets/inspection-gates.js"></script>'
                '<script type="module" src="/assets/app.js">',
            ),
            content_type="text/html",
        )

    async def asset(self, request):
        if request.match_info["path"] in {"inspection-gates.js", "inspection-checks.js"}:
            return web.FileResponse(Path(__file__).with_name(request.match_info["path"]))
        return await super().asset(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path)
    args = parser.parse_args()
    args.port, args.no_open, args.episodes, args.fps = 0, True, 0, 20
    with TemporaryDirectory(prefix="gradlab-inspection-") as temporary:
        root = Path(temporary)
        write_bundle(root)
        loader = SelectionLoader(args, root)
        host = SelectionHost(loader, initial_route=ROUTE)
        try:
            asyncio.run(InspectionPlayer(host, args, catalog=SelectionCatalog()).run())
        finally:
            loader.ready.set()
            host.stop()


if __name__ == "__main__":
    main()
