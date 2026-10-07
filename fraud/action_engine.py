"""Action recommender: map risk to a recommended response.

These are decision-support recommendations for a fraud analyst. Nothing here blocks a
payment or freezes an account by itself; that requires integration into an authorised
banking system with its own controls.
"""

DISCLAIMER = ("Decision support only: recommendations are for a fraud analyst. No payment is "
              "blocked and no account is frozen unless this is integrated into an authorised "
              "banking system.")

TXN_ACTIONS = {
    "LOW": ("ALLOW", "Transaction appears consistent with normal behavior."),
    "MEDIUM": ("MONITOR", "Continue monitoring account behavior."),
    "HIGH": ("STEP-UP AUTHENTICATION",
             "Trigger OTP/biometric verification and additional verification."),
    "CRITICAL": ("BLOCK + INVESTIGATE",
                 "Temporarily block transaction and investigate connected entities."),
}
STATUS = {"LOW": "NORMAL", "MEDIUM": "REVIEW", "HIGH": "STEP-UP", "CRITICAL": "BLOCK"}

ACCOUNT_ACTIONS = {
    "LOW": ("NO ACTION", "Account behaviour is consistent with its profile."),
    "MEDIUM": ("MONITOR", "Add to watch-list; review if new risk signals appear."),
    "HIGH": ("STEP-UP + RESTRICT", "Require re-verification and limit outgoing transfers "
                                   "until the customer confirms recent activity."),
    "CRITICAL": ("FREEZE + INVESTIGATE", "Temporarily freeze outgoing payments and open a "
                                         "case covering all linked accounts and devices."),
}


def txn_action(band):
    return TXN_ACTIONS[band][0]


def txn_rationale(band):
    return TXN_ACTIONS[band][1]


def account_action(band):
    return ACCOUNT_ACTIONS[band][0]


def ring_actions(ring):
    """Ordered response playbook for a detected ring."""
    risk = ring.get("ring_risk", 0)
    steps = []
    if ring.get("mules"):
        steps.append(("Freeze suspicious pathway",
                      f"Block outgoing payouts from beneficiary/mule account(s) "
                      f"{', '.join(ring['mules'])} and hold inbound transfers to them."))
    if ring.get("collusive_merchants"):
        steps.append(("Suspend merchant settlement",
                      f"Hold settlement for {', '.join(ring['collusive_merchants'])} pending "
                      f"merchant review."))
    members = [m for m in ring.get("members", []) if m not in ring.get("mules", [])]
    steps.append(("Investigate linked accounts",
                  f"Open one case for {len(members)} member accounts: "
                  f"{', '.join(members[:10])}{' …' if len(members) > 10 else ''}."))
    if ring.get("devices"):
        steps.append(("Review shared devices",
                      f"Check device fingerprints {', '.join(ring['devices'][:6])} for other "
                      f"accounts; block the device IDs for new sign-ups."))
    if ring.get("beneficiaries"):
        steps.append(("Review beneficiaries",
                      f"Verify beneficiaries {', '.join(ring['beneficiaries'][:6])}; flag them "
                      f"for confirmation-of-payee warnings."))
    steps.append(("Escalate to fraud analyst",
                  "Assign to the financial-crime team; file a suspicious transaction report if "
                  "the investigation confirms coordinated fraud."))
    headline = ("BLOCK + INVESTIGATE (critical ring)" if risk >= 81 else
                "STEP-UP + INVESTIGATE" if risk >= 61 else "MONITOR RING")
    return headline, steps
