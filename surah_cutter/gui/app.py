"""GUI entry point.

Mounts the Phase-1 dashboard and the Phase-5 review screen in a
``QStackedWidget``: a surah's "Review" (or double-click) swaps in
:class:`~surah_cutter.gui.review_view.ReviewView`; its "Back" returns to the
dashboard and refreshes surah statuses.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    # shortcut: defer PySide6 import so the module is light to import/test.
    try:
        from PySide6.QtWidgets import QApplication, QStackedWidget
    except ImportError:
        print("PySide6 is required for the GUI. Install with: pip install -e .")
        return 1

    from surah_cutter.gui.dashboard_view import DashboardView
    from surah_cutter.gui.review_view import ReviewView

    app = QApplication(argv if argv is not None else sys.argv)

    stack = QStackedWidget()
    dashboard = DashboardView()
    stack.addWidget(dashboard)
    stack.setCurrentWidget(dashboard)

    # One review screen at a time; tracked so we can dispose it on return.
    state: dict[str, object] = {"review": None}

    def open_review(db_path: str, surah: int, audio_path: str) -> None:
        review = ReviewView(db_path, surah, audio_path)
        review.back_requested.connect(close_review)
        state["review"] = review
        stack.addWidget(review)
        stack.setCurrentWidget(review)

    def close_review() -> None:
        review = state.get("review")
        if review is not None:
            stack.setCurrentWidget(dashboard)
            stack.removeWidget(review)  # type: ignore[arg-type]
            review.deleteLater()  # type: ignore[union-attr]
            state["review"] = None
        dashboard.refresh_statuses()

    dashboard.review_requested.connect(open_review)

    stack.setWindowTitle("surah-cutter")
    stack.resize(900, 760)
    stack.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
