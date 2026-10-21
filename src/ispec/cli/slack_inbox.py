"""Local-only review of received Slack text; no network or comment writes."""

import json

from ispec.assistant.slack_inbox import SlackInbox


def register(subparsers):
    parser = subparsers.add_parser("inbox", help="Review locally retained Slack messages; never sends")
    parser.add_argument("--database", help="Assistant DB override; defaults to configured assistant DB")
    actions = parser.add_subparsers(dest="inbox_action", required=True)
    listing = actions.add_parser("list", help="List unreviewed messages, newest first")
    listing.add_argument("--channel")
    listing.add_argument("--all", action="store_true", help="Include reviewed messages")
    listing.add_argument("--before-id", type=int)
    listing.add_argument("--limit", type=int, default=50)
    show = actions.add_parser("show")
    show.add_argument("id", type=int)
    review = actions.add_parser("review", help="Mark reviewed, NOT approved for a write")
    review.add_argument("id", type=int)
    review.add_argument("--by", required=True, help="Local review attribution; not a Slack permission grant")


def dispatch(args):
    inbox = SlackInbox(file=args.database)
    try:
        if args.inbox_action == "review":
            inbox.review(args.id, actor=args.by)
            result = {"reviewed": True, "id": args.id, "write_authorized": False}
        elif args.inbox_action == "show":
            result = inbox.records(row_id=args.id, unreviewed=False)
        else:
            result = inbox.records(channel=args.channel, unreviewed=not args.all,
                                   before_id=args.before_id, limit=args.limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, default=str, indent=2))
