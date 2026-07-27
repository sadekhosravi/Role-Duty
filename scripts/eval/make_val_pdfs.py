"""Build the validation corpus: five fictional Role & Duty PDFs in data/val/raw.

These are deliberately adversarial. They share a world — one fire department
covers the airport, the school, the library and the data centre — so answering
some questions requires reading two documents, and several traps only spring if
the model conflates similarly-named roles across sites:

  - THREE roles are called "... Duty Manager" (Airside, Terminal, Site) with
    different authority; two more are "... Lead" (Ramp, Shift).
  - Rank does not imply authority: a Battalion Chief outranks the Fire Marshal
    on scene but cannot close an occupancy; a Principal cannot overrule the Fire
    Marshal; an Airside Duty Manager cannot lift the Safety Officer's stop-work.
  - Numeric ladders (25 litres, $10/$50, 30 days, 5 minutes, Severity-1 vs -2)
    reward exact retrieval and punish plausible guessing.
  - The same task has different owners at different sites: the school's annual
    fire inspection is booked by the Campus Safety Coordinator, the library's by
    the Facilities Supervisor.
  - Some facts are simply absent (staffing per engine, pay rates), so a model
    that always produces an answer scores worse than one that declines.

Run:  uv run --with reportlab python scripts/eval/make_val_pdfs.py
Output is deterministic — regenerating overwrites the same five files.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "val" / "raw"

DISCLAIMER = (
    "This is a fictional, simplified reference created for testing a retrieval "
    "system. It describes no real organisation and is not operational guidance."
)

# --- The corpus ----------------------------------------------------------------
#
# Each role: title, unit, reports_to, summary, duties, out_of_scope, escalation.

DOCS = [
    {
        "file": "sample_role_duties_airport.pdf",
        "title": "Northgate Municipal Airport - Ground Operations Role & Duty Reference",
        "intro": [
            "Sample dataset for RAG testing. Fictional regional airport, five roles, "
            "their duties, boundaries, and escalation paths.",
            "Each entry follows the same structure: Reports To, Summary, Key "
            "Responsibilities, Out of Scope, and Escalation Path. " + DISCLAIMER,
        ],
        "roles": [
            {
                "title": "Ramp Agent",
                "unit": "Ground Operations",
                "reports_to": "Ramp Lead",
                "summary": (
                    "Handles aircraft on the stand during turnaround - marshalling, "
                    "baggage, and ground service equipment - under the direction of the "
                    "Ramp Lead."
                ),
                "duties": [
                    "Marshal arriving aircraft onto the stand and place chocks and cones "
                    "once the engines are shut down.",
                    "Load and unload baggage, mail, and cargo, and reconcile bag counts "
                    "against the loading instruction report.",
                    "Operate belt loaders, baggage tractors, and ground power units for "
                    "which the agent holds a current equipment endorsement.",
                    "Walk the stand for foreign object debris (FOD) before and after every "
                    "aircraft movement.",
                    "Report any contact between equipment and an aircraft immediately, "
                    "however minor it appears.",
                    "Complete the turnaround checklist and hand it to the Ramp Lead at the "
                    "end of the turn.",
                ],
                "out_of_scope": [
                    "Refuelling aircraft - fuelling at Northgate is performed solely by "
                    "Northgate Fuel Services under contract.",
                    "Operating a pushback tug without a current Type-B tug certification.",
                    "Signing the aircraft weight and balance sheet - this is the airline's "
                    "load controller.",
                    "Entering the movement area (taxiways or runway) without radio clearance "
                    "from Air Traffic Control relayed by the Ramp Lead.",
                ],
                "escalation": (
                    "Report FOD, a fuel leak, or any equipment-to-aircraft contact to the "
                    "Ramp Lead immediately and stop work on that stand. If the Ramp Lead is "
                    "not reachable within two minutes and an aircraft is about to move, "
                    "contact the Airside Duty Manager directly on the airside channel. "
                    "Personal injury is reported to the Ramp Lead and the Airside Safety "
                    "Officer."
                ),
            },
            {
                "title": "Ramp Lead",
                "unit": "Ground Operations",
                "reports_to": "Airside Duty Manager",
                "summary": (
                    "Runs the ramp crew for a shift: allocates agents to stands, verifies "
                    "safety-critical steps, and owns turnaround performance for the shift."
                ),
                "duties": [
                    "Assign Ramp Agents to stands at the start of each shift and rebalance "
                    "crews as the schedule moves.",
                    "Hold the pre-shift safety briefing and record attendance.",
                    "Verify chocks, cones, and equipment clearance before authorising a "
                    "pushback.",
                    "Track turnaround timings and report delays attributable to ground "
                    "handling.",
                    "Authorise swapping a defective ground service unit for a spare and "
                    "raise the maintenance ticket.",
                    "Manage a spill of 25 litres or less using the stand spill kit, and log "
                    "it in the airside occurrence system within 4 hours.",
                ],
                "out_of_scope": [
                    "Closing or reopening a stand, taxiway, or runway - this rests with the "
                    "Airside Duty Manager.",
                    "Issuing formal disciplinary action, which is handled by the Airside "
                    "Duty Manager with Human Resources.",
                    "Approving more than 2 hours of overtime per agent per shift.",
                    "Lifting a stop-work order issued by the Airside Safety Officer.",
                ],
                "escalation": (
                    "Escalate stand conflicts, equipment shortages that will delay a "
                    "departure, and any injury to the Airside Duty Manager. A spill larger "
                    "than 25 litres is not a Ramp Lead matter: hand it to the Airside Duty "
                    "Manager at once."
                ),
            },
            {
                "title": "Airside Safety Officer",
                "unit": "Safety & Compliance",
                "reports_to": "Head of Compliance",
                "summary": (
                    "Independent safety assurance for the airside. Inspects, audits, and "
                    "investigates, and holds stop-work authority that sits outside the "
                    "operational chain of command."
                ),
                "duties": [
                    "Conduct the daily airside inspection and the weekly FOD walk audit.",
                    "May stop any airside operation immediately where an unsafe condition is "
                    "observed, without prior approval from operations management.",
                    "Investigate ramp incidents and near-misses and publish findings.",
                    "Maintain the confidential safety reporting system and brief trends "
                    "monthly.",
                    "Advise the Airside Duty Manager on the safety implications of stand and "
                    "taxiway closures.",
                ],
                "out_of_scope": [
                    "Directing turnaround priorities or allocating stands - the Safety "
                    "Officer advises operations but does not run them.",
                    "Assigning or rescheduling ramp crews.",
                    "Reversing a stand or taxiway closure made by the Airside Duty Manager.",
                    "Making a regulatory notification directly to the Civil Aviation "
                    "Authority.",
                ],
                "escalation": (
                    "A stop-work order issued by the Airside Safety Officer may be lifted "
                    "only by the Head of Compliance, after the officer's findings have been "
                    "closed out; no operational manager, including the Airside Duty Manager "
                    "or the Director of Airport Operations, may restart the work in the "
                    "meantime. Reportable occurrences are passed to the Head of Compliance, "
                    "who notifies the Civil Aviation Authority within 24 hours."
                ),
            },
            {
                "title": "Airside Duty Manager",
                "unit": "Airside Operations",
                "reports_to": "Director of Airport Operations",
                "summary": (
                    "Holds operational authority for everything beyond the terminal doors "
                    "during an assigned shift, including the apron, taxiways, and the "
                    "airport's first-line emergency response."
                ),
                "duties": [
                    "Close and reopen stands, taxiways, and the runway, and notify Air "
                    "Traffic Control of each change.",
                    "Declare a Local Standby (Emergency Plan Level 1) and alert the "
                    "responding agencies.",
                    "Authorise the airside response to a fuel spill and appoint the airport "
                    "spill controller.",
                    "Coordinate airside works, vehicle permits, and contractor escorts.",
                    "Chair the post-incident airside debrief and sign the shift handover.",
                ],
                "out_of_scope": [
                    "Declaring a Full Emergency (Emergency Plan Level 2) - reserved to the "
                    "Director of Airport Operations.",
                    "Retaining or assuming incident command once Cedar Ridge Regional Fire & "
                    "Rescue units arrive on scene; command passes to their Incident "
                    "Commander under the Airport Mutual Aid Annex, and the Airside Duty "
                    "Manager continues in a supporting liaison role only.",
                    "Lifting a stop-work order issued by the Airside Safety Officer.",
                    "Any decision inside the terminal building, including gate reassignment "
                    "and passenger communications.",
                ],
                "escalation": (
                    "A fuel spill larger than 25 litres, any spill reaching a drain, and any "
                    "fire or smoke on the apron are called immediately to Cedar Ridge "
                    "Regional Fire & Rescue on the Station 4 direct line, with the Airside "
                    "Safety Officer notified in parallel. Escalate to the Director of "
                    "Airport Operations for anything requiring a Full Emergency, an aircraft "
                    "accident, or a closure expected to last beyond the shift."
                ),
            },
            {
                "title": "Terminal Duty Manager",
                "unit": "Terminal Operations",
                "reports_to": "Director of Airport Operations",
                "summary": (
                    "Runs the passenger terminal for a shift - flow, gates, and "
                    "communications. Despite the similar title, this role holds no airside "
                    "authority whatsoever."
                ),
                "duties": [
                    "Manage passenger flow through check-in, security queueing, and "
                    "boarding.",
                    "Reassign gates within the terminal and update the flight information "
                    "displays.",
                    "Approve delay and disruption announcements and welfare provision for "
                    "waiting passengers.",
                    "Own the lost and mishandled baggage escalation path with the handling "
                    "airlines.",
                    "Coordinate cleaning, retail, and terminal facilities issues.",
                ],
                "out_of_scope": [
                    "Any decision affecting the apron, taxiways, or runway - including "
                    "closing a stand or a taxiway, which is the Airside Duty Manager's "
                    "decision alone.",
                    "Authorising an aircraft to return to stand.",
                    "Directing ramp crews or ground service equipment.",
                    "Declaring any level of the airport emergency plan.",
                ],
                "escalation": (
                    "Escalate to the Director of Airport Operations for terminal closures, "
                    "mass disruption, or a media enquiry. Security incidents go to the "
                    "Airport Police Post. Anything originating airside is handed to the "
                    "Airside Duty Manager rather than actioned."
                ),
            },
        ],
    },
    {
        "file": "sample_role_duties_fire.pdf",
        "title": "Cedar Ridge Regional Fire & Rescue - Role & Duty Reference Guide",
        "intro": [
            "Sample dataset for RAG testing. Fictional combination fire department "
            "covering the Cedar Ridge district, five roles, their duties, boundaries, "
            "and escalation paths.",
            "Cedar Ridge Station 4 provides first response to Northgate Municipal "
            "Airport and the Northgate Industrial Park, including the Blackwater Data "
            "Centre, under the Airport Mutual Aid Annex. " + DISCLAIMER,
        ],
        "roles": [
            {
                "title": "Firefighter/EMT",
                "unit": "Suppression",
                "reports_to": "Company Officer (Lieutenant)",
                "summary": (
                    "Front-line member of an engine or ladder company performing "
                    "suppression, rescue, and basic life support under the direction of the "
                    "Company Officer."
                ),
                "duties": [
                    "Perform fire attack, search, ventilation, and extrication as assigned "
                    "on the fireground.",
                    "Provide basic life support patient care and assist the transporting "
                    "ambulance crew.",
                    "Complete daily apparatus, equipment, and self-contained breathing "
                    "apparatus checks.",
                    "Maintain hydrants and pre-incident survey information in the assigned "
                    "district.",
                    "Support public education events under the direction of the Fire & Life "
                    "Safety Educator.",
                ],
                "out_of_scope": [
                    "Administering advanced life support medications or performing advanced "
                    "airway procedures - restricted to a certified Paramedic.",
                    "Determining the cause and origin of a fire beyond recording preliminary "
                    "observations.",
                    "Releasing any information about an incident to the media or on social "
                    "media.",
                    "Entering an atmosphere immediately dangerous to life or health without a "
                    "partner and a charged line.",
                ],
                "escalation": (
                    "Report changing fire conditions, a structural concern, or a lost or "
                    "trapped member to the Company Officer by radio immediately using the "
                    "emergency traffic tone. Injuries and exposures are reported to the "
                    "Company Officer and the department Safety Officer before the end of "
                    "shift."
                ),
            },
            {
                "title": "Company Officer (Lieutenant)",
                "unit": "Suppression",
                "reports_to": "Battalion Chief",
                "summary": (
                    "Commands a single engine or ladder company, and is the first-arriving "
                    "officer at most incidents in the district."
                ),
                "duties": [
                    "Assume initial Incident Command on arrival, transmit an on-scene "
                    "report, and hold command until a Battalion Chief arrives and formally "
                    "announces the transfer of command over the radio.",
                    "Assign tactical objectives to the company and account for every member "
                    "at each benchmark.",
                    "Conduct the 360-degree size-up and set the initial incident action "
                    "plan.",
                    "Complete the incident report and the company's portion of the "
                    "post-incident review.",
                    "Supervise company training, station duties, and daily readiness.",
                ],
                "out_of_scope": [
                    "Declaring a second alarm or any additional alarm - this is the "
                    "Battalion Chief's decision.",
                    "Committing mutual aid resources from a neighbouring department.",
                    "Closing a public roadway, which is requested through the police "
                    "department.",
                    "Issuing a notice of violation or any fire-code enforcement action.",
                ],
                "escalation": (
                    "Request the Battalion Chief for any working fire, any incident "
                    "requiring more than the assigned companies, or any firefighter injury. "
                    "Command is not passed by arrival alone: the Company Officer remains "
                    "Incident Commander until the Battalion Chief states on the radio that "
                    "command has transferred."
                ),
            },
            {
                "title": "Battalion Chief (Shift Commander)",
                "unit": "Operations",
                "reports_to": "Deputy Chief of Operations",
                "summary": (
                    "Commands the shift across the district, assumes Incident Command at "
                    "working incidents, and is the department's senior officer on scene."
                ),
                "duties": [
                    "Assume Incident Command at working incidents by announcing the transfer "
                    "of command over the radio.",
                    "Request additional alarms, specialist resources, and mutual aid.",
                    "Serve as Incident Commander at Northgate Municipal Airport and at the "
                    "Blackwater Data Centre once Cedar Ridge units arrive on scene, under "
                    "the Airport Mutual Aid Annex.",
                    "Manage shift staffing, apparatus availability, and station coverage.",
                    "Conduct post-incident analysis for working incidents in the battalion.",
                ],
                "out_of_scope": [
                    "Fire-code enforcement of any kind - issuing a notice of violation, "
                    "citing an owner, or ordering an occupancy closed rests solely with the "
                    "Fire Marshal, regardless of rank on scene.",
                    "Hiring, promotion, or formal discipline, which sit with the Deputy "
                    "Chief of Operations and the Fire Chief.",
                    "Making statements to the media beyond confirmed incident facts.",
                    "Operating at a hazardous materials incident beyond the Operations level "
                    "with department resources alone.",
                ],
                "escalation": (
                    "Escalate to the Deputy Chief of Operations for a multiple-alarm "
                    "incident, a line-of-duty injury, or an incident expected to run past "
                    "the shift. Hazardous materials incidents beyond the Operations level "
                    "are escalated by requesting Regional Hazmat Team 7 through County "
                    "Dispatch. Suspicious fires are referred to the Fire Marshal for cause "
                    "and origin investigation."
                ),
            },
            {
                "title": "Fire Marshal",
                "unit": "Fire Prevention Bureau",
                "reports_to": "Fire Chief",
                "summary": (
                    "Holds the department's fire-code enforcement and fire-investigation "
                    "authority. Works in prevention rather than suppression."
                ),
                "duties": [
                    "Conduct the annual fire inspection of schools, libraries, and other "
                    "public assembly occupancies in the district.",
                    "Issue notices of violation, set correction deadlines, and order an "
                    "occupancy closed where an imminent hazard to life exists.",
                    "Review construction and renovation plans for fire-code compliance.",
                    "Determine cause and origin for fires referred for investigation.",
                    "Approve the posted occupant load for assembly spaces.",
                ],
                "out_of_scope": [
                    "Assuming Incident Command at an active fire - command rests with the "
                    "Battalion Chief or the first-arriving Company Officer.",
                    "Directing suppression tactics on the fireground.",
                    "Delivering public education programmes, which are run by the Fire & "
                    "Life Safety Educator.",
                    "Waiving a code requirement without a documented alternative-means "
                    "review approved by the Fire Chief.",
                ],
                "escalation": (
                    "An order to close an occupancy takes effect immediately and stands "
                    "until the Fire Marshal is satisfied the hazard is corrected; it is not "
                    "subject to override by the occupancy's own management, and an appeal "
                    "runs to the Fire Chief and then the district Board of Appeals. "
                    "Suspected arson is referred to the police department and the Fire "
                    "Chief."
                ),
            },
            {
                "title": "Fire & Life Safety Educator",
                "unit": "Fire Prevention Bureau",
                "reports_to": "Fire Marshal",
                "summary": (
                    "Civilian member who delivers the department's public education "
                    "programme in schools, libraries, and community venues."
                ),
                "duties": [
                    "Deliver fire-safety and evacuation programmes in district schools and "
                    "libraries.",
                    "Run the residential smoke-alarm installation programme.",
                    "Host station tours and community events.",
                    "Produce public safety materials and campaign content for review by the "
                    "Fire Marshal.",
                    "Maintain records of programmes delivered and audiences reached.",
                ],
                "out_of_scope": [
                    "Conducting fire inspections or signing off on any violation - the "
                    "Educator has no enforcement authority.",
                    "Scheduling or performing the annual inspection of any occupancy.",
                    "Entering an active incident scene.",
                    "Advising on code compliance for a specific building, which is referred "
                    "to the Fire Marshal.",
                ],
                "escalation": (
                    "Hazards noticed while visiting a school or library - blocked exits, "
                    "disabled alarms, overcrowded rooms - are reported to the Fire Marshal, "
                    "who decides whether an inspection or enforcement action follows. The "
                    "Educator does not raise these with the site's management directly."
                ),
            },
        ],
    },
    {
        "file": "sample_role_duties_school.pdf",
        "title": "Fairview Independent School District - Westbrook Campus Role & Duty Reference",
        "intro": [
            "Sample dataset for RAG testing. Fictional secondary campus, five roles, "
            "their duties, boundaries, and escalation paths.",
            "Westbrook Campus sits in the Cedar Ridge fire district and is inspected "
            "annually by the Cedar Ridge Fire Marshal. " + DISCLAIMER,
        ],
        "roles": [
            {
                "title": "Classroom Teacher",
                "unit": "Instruction",
                "reports_to": "Assistant Principal",
                "summary": (
                    "Delivers instruction to assigned classes and is responsible for the "
                    "supervision and safety of students in their care."
                ),
                "duties": [
                    "Plan and deliver lessons against the district curriculum and record "
                    "assessment outcomes.",
                    "Take attendance every period and report discrepancies to the Front "
                    "Office Manager.",
                    "Supervise students during class, transitions, and assigned duty "
                    "periods.",
                    "Lead their class to the assigned assembly point during a drill or "
                    "evacuation and take a headcount.",
                    "Contact parents about academic progress and routine behaviour concerns.",
                ],
                "out_of_scope": [
                    "Administering any medication, including over-the-counter medication - "
                    "this is the School Nurse.",
                    "Releasing a student to any adult, including a parent, during an "
                    "emergency or reunification.",
                    "Issuing a suspension or any formal disciplinary sanction.",
                    "Initiating a lockdown or an evacuation.",
                ],
                "escalation": (
                    "Report a medical concern to the School Nurse and a safety or behaviour "
                    "emergency to the Front Office by intercom, which alerts the Campus "
                    "Safety Coordinator. Suspected abuse or neglect is reported the same day "
                    "to the Principal, who makes the statutory report."
                ),
            },
            {
                "title": "Campus Safety Coordinator",
                "unit": "Campus Operations",
                "reports_to": "Principal",
                "summary": (
                    "Owns the campus emergency operations plan, the drill programme, and "
                    "the day-to-day safety and security of the site."
                ),
                "duties": [
                    "Maintain the campus emergency operations plan and the site maps issued "
                    "to responders.",
                    "Run one fire drill every month and two lockdown drills every school "
                    "year, and record the results.",
                    "Schedule the annual fire inspection with the Cedar Ridge Fire Marshal "
                    "and walk the campus with the inspector.",
                    "Initiate a lockdown when a threat is identified, and announce it "
                    "through the Front Office.",
                    "Manage access control, visitor screening rules, and the camera system.",
                ],
                "out_of_scope": [
                    "Releasing students to parents or guardians during an emergency or "
                    "reunification - only the Principal may authorise release.",
                    "Ordering a campus evacuation, which is the Principal's decision.",
                    "Issuing student discipline.",
                    "Speaking to the media or to parents on behalf of the campus.",
                ],
                "escalation": (
                    "Notify the Principal immediately on initiating a lockdown; the "
                    "Coordinator may start one without prior approval but cannot end it "
                    "without the Principal. Structural, fire-protection, or exiting hazards "
                    "are reported to the Principal and to the Cedar Ridge Fire Marshal."
                ),
            },
            {
                "title": "School Nurse",
                "unit": "Health Services",
                "reports_to": (
                    "Principal for day-to-day administration, and District Health Services "
                    "for clinical practice"
                ),
                "summary": (
                    "Provides health services to students on campus under a dual reporting "
                    "line: administratively to the Principal, clinically to District Health "
                    "Services."
                ),
                "duties": [
                    "Administer prescribed medication where a current physician order and "
                    "signed parent consent are both on file.",
                    "Maintain individual health care plans for students with chronic "
                    "conditions.",
                    "Provide first aid and decide whether a student returns to class, goes "
                    "home, or needs emergency transport.",
                    "Exclude a student from campus for a communicable condition under "
                    "district health protocol.",
                    "Track immunisation compliance and report it to District Health "
                    "Services.",
                ],
                "out_of_scope": [
                    "Diagnosing a condition or recommending a specific treatment.",
                    "Changing a prescribed dose or schedule, even at a parent's request - "
                    "this requires a new physician order.",
                    "Sharing a student's health information with staff beyond what a "
                    "specific role needs to keep the student safe.",
                    "Overruling a clinical protocol issued by District Health Services.",
                ],
                "escalation": (
                    "Call emergency medical services directly for a life-threatening "
                    "presentation and notify the Principal immediately afterwards. Clinical "
                    "questions and protocol conflicts go to District Health Services, not to "
                    "the Principal; a disagreement between an administrative instruction and "
                    "a clinical protocol is resolved in favour of the clinical protocol and "
                    "escalated to District Health Services the same day."
                ),
            },
            {
                "title": "Front Office Manager",
                "unit": "Campus Operations",
                "reports_to": "Principal",
                "summary": (
                    "Runs the campus front office: visitors, attendance records, emergency "
                    "contact data, and the public address system."
                ),
                "duties": [
                    "Check in every visitor against photo identification and issue a badge.",
                    "Maintain daily attendance records and the emergency contact and "
                    "authorised-pickup lists.",
                    "Make announcements over the public address system, including drill and "
                    "lockdown announcements at the direction of the Campus Safety "
                    "Coordinator or the Principal.",
                    "Route calls and enquiries, and log parent concerns for the Principal.",
                    "Maintain the student sign-out log during normal operations.",
                ],
                "out_of_scope": [
                    "Deciding whether an adult who is not on the authorised-pickup list may "
                    "collect a student - referred to the Principal.",
                    "Initiating a lockdown or evacuation on their own authority.",
                    "Discussing a student's records or health information with anyone other "
                    "than an authorised staff member or guardian.",
                    "Admitting a visitor who declines screening.",
                ],
                "escalation": (
                    "Alert the Campus Safety Coordinator immediately for an unauthorised "
                    "visitor, a refusal to leave, or a custody dispute at the front counter, "
                    "and the Principal in parallel. During a lockdown the office phone line "
                    "is kept clear for responders."
                ),
            },
            {
                "title": "Principal",
                "unit": "Campus Leadership",
                "reports_to": "Superintendent",
                "summary": (
                    "Holds final authority for the campus - instruction, staff, discipline, "
                    "and emergency decisions."
                ),
                "duties": [
                    "Order a campus evacuation, shelter-in-place, or reunification, and "
                    "authorise the release of students to guardians.",
                    "Make final student discipline decisions, including suspension "
                    "recommendations.",
                    "Supervise and evaluate campus staff.",
                    "Make the statutory report where abuse or neglect is suspected.",
                    "Approve campus expenditure within the delegated limit.",
                ],
                "out_of_scope": [
                    "Overriding an order from the Cedar Ridge Fire Marshal to close all or "
                    "part of the campus - the order stands until the Fire Marshal lifts it, "
                    "and any appeal runs through the Superintendent to the Fire Chief.",
                    "Setting district-wide policy or curriculum.",
                    "Approving a contract above $10,000, which requires the Superintendent.",
                    "Overruling a District Health Services clinical protocol.",
                ],
                "escalation": (
                    "Notify the Superintendent for any campus closure, serious injury, "
                    "police incident, or media enquiry. Where the Fire Marshal has closed a "
                    "space, the Principal arranges alternative accommodation and reports the "
                    "closure to the Superintendent rather than contesting it on site."
                ),
            },
        ],
    },
    {
        "file": "sample_role_duties_library.pdf",
        "title": "Harbor Point Public Library - Role & Duty Reference Guide",
        "intro": [
            "Sample dataset for RAG testing. Fictional branch library, five roles, their "
            "duties, boundaries, and escalation paths.",
            "Harbor Point is a branch of the county library system and sits in the Cedar "
            "Ridge fire district. " + DISCLAIMER,
        ],
        "roles": [
            {
                "title": "Circulation Assistant",
                "unit": "Circulation",
                "reports_to": "Branch Manager",
                "summary": (
                    "Staffs the circulation desk: borrowing, returns, registrations, and "
                    "routine account questions."
                ),
                "duties": [
                    "Issue, renew, and check in loans, and register new borrowers against "
                    "proof of address.",
                    "Collect fines and replacement charges and reconcile the till at close.",
                    "Waive fines and fees up to $10 per account without further approval.",
                    "Shelve returned stock and run the daily holds and reservations list.",
                    "Refer research questions to the Reference Librarian rather than "
                    "answering them at the desk.",
                ],
                "out_of_scope": [
                    "Waiving more than $10 on an account.",
                    "Suspending or restoring a borrower's privileges.",
                    "Giving out any borrower's account details, including to a family "
                    "member.",
                    "Approving a fee refund to a payment card, which is processed by the "
                    "Branch Manager.",
                ],
                "escalation": (
                    "Refer waiver requests above $10 to the Branch Manager. Disruptive "
                    "behaviour, a suspected theft, or a dispute at the desk is escalated to "
                    "the Branch Manager immediately; staff do not detain anyone."
                ),
            },
            {
                "title": "Reference Librarian",
                "unit": "Public Services",
                "reports_to": "Branch Manager",
                "summary": (
                    "Provides research help, information literacy instruction, and access "
                    "to the branch's reference collection and databases."
                ),
                "duties": [
                    "Conduct reference interviews and help patrons locate authoritative "
                    "sources.",
                    "Teach scheduled classes on research skills, databases, and basic "
                    "digital literacy.",
                    "Maintain the reference and local history collections and recommend "
                    "purchases.",
                    "Support patrons using public computers and the online catalogue.",
                    "Refer patrons needing legal, medical, or tax help to the relevant "
                    "self-help collections, official forms, and outside services.",
                ],
                "out_of_scope": [
                    "Giving legal, medical, or tax advice, or interpreting what a form, "
                    "notice, or court document means for a patron's situation - the "
                    "librarian may locate the correct form and explain where to get advice, "
                    "and nothing further.",
                    "Completing a form on a patron's behalf or witnessing a signature.",
                    "Recommending a specific attorney, clinician, or preparer.",
                    "Overriding a collection development decision made by the Branch "
                    "Manager.",
                ],
                "escalation": (
                    "Refer patrons who need advice rather than information to the listed "
                    "legal aid, health, and taxpayer assistance services. Escalate a "
                    "distressed patron, a safeguarding concern, or a request for records to "
                    "the Branch Manager."
                ),
            },
            {
                "title": "Youth Services Librarian",
                "unit": "Public Services",
                "reports_to": "Branch Manager",
                "summary": (
                    "Runs programming and collections for children and teenagers and is the "
                    "branch's first contact for concerns about a young patron."
                ),
                "duties": [
                    "Plan and deliver story times, holiday programmes, and the summer "
                    "reading challenge.",
                    "Select and maintain the children's and young adult collections.",
                    "Support school class visits and liaise with local campuses.",
                    "Apply the unattended child rule: a child under 9 must be accompanied by "
                    "a responsible person aged 14 or over.",
                    "Host the Cedar Ridge Fire & Life Safety Educator's school-holiday "
                    "safety sessions.",
                ],
                "out_of_scope": [
                    "Allowing an unaccompanied child under 9 to remain in the branch - the "
                    "Branch Manager must be notified without delay.",
                    "Making a report to child protective services directly; concerns are "
                    "reported to the Branch Manager, who makes the report.",
                    "Transporting a child or taking a child outside the building.",
                    "Excluding a young patron from the branch, which only the Branch Manager "
                    "may do.",
                ],
                "escalation": (
                    "Notify the Branch Manager immediately about an unattended child, a "
                    "safeguarding concern, or an incident during a programme. If a child is "
                    "still unaccompanied at closing time, the Branch Manager contacts the "
                    "police non-emergency line and a second staff member remains present."
                ),
            },
            {
                "title": "Facilities Supervisor",
                "unit": "Facilities",
                "reports_to": "Branch Manager",
                "summary": (
                    "Keeps the building, its systems, and its contractors running, and is "
                    "the branch's contact for inspections and building compliance."
                ),
                "duties": [
                    "Schedule the annual fire inspection with the Cedar Ridge Fire Marshal "
                    "and accompany the inspector.",
                    "Maintain fire extinguishers, emergency lighting, and exit routes, and "
                    "keep the testing log.",
                    "Manage cleaning, maintenance, and contractor access.",
                    "Shut down a failing system or close a single affected area - a "
                    "restroom, a wing, the meeting room - where there is a hazard.",
                    "Correct items on a notice of violation within the deadline the notice "
                    "sets.",
                ],
                "out_of_scope": [
                    "Closing the branch to the public - that decision belongs to the Branch "
                    "Manager.",
                    "Contesting or negotiating the contents of a fire-code notice of "
                    "violation.",
                    "Approving capital works or a contract above the branch's delegated "
                    "limit.",
                    "Directing public-service staff or programming.",
                ],
                "escalation": (
                    "Report a hazard that affects the whole building - loss of water, a "
                    "failed fire alarm panel, flooding - to the Branch Manager at once with "
                    "a recommendation. A notice of violation received from the Fire Marshal "
                    "is passed to the Branch Manager the same day and copied to the Library "
                    "Director."
                ),
            },
            {
                "title": "Branch Manager",
                "unit": "Branch Leadership",
                "reports_to": "Library Director",
                "summary": (
                    "Holds day-to-day authority for the branch: staff, service, spending "
                    "within delegation, and patron conduct."
                ),
                "duties": [
                    "Supervise branch staff, set rosters, and handle first-line performance "
                    "matters.",
                    "Waive fines and fees up to $50 per account.",
                    "Suspend a patron's borrowing and access privileges for up to 30 days "
                    "for a conduct breach.",
                    "Close the branch to the public for up to one day for a hazard, "
                    "emergency, or staffing failure.",
                    "Make reports to child protective services and to the police on the "
                    "branch's behalf.",
                ],
                "out_of_scope": [
                    "Waiving more than $50 on an account - referred to the Library Director.",
                    "Suspending a patron for longer than 30 days, which requires the Library "
                    "Director.",
                    "Closing the branch for more than one day, which requires the Library "
                    "Director.",
                    "Overruling, appealing, or negotiating an order or notice issued by the "
                    "Cedar Ridge Fire Marshal.",
                ],
                "escalation": (
                    "Escalate to the Library Director for waivers above $50, suspensions "
                    "beyond 30 days, closures beyond one day, and any media enquiry. A fire "
                    "code notice of violation is acknowledged, assigned to the Facilities "
                    "Supervisor for correction within the stated deadline, and reported to "
                    "the Library Director; the branch does not dispute it locally."
                ),
            },
        ],
    },
    {
        "file": "sample_role_duties_datacenter.pdf",
        "title": "Blackwater Data Centre - Site Operations Role & Duty Reference",
        "intro": [
            "Sample dataset for RAG testing. Fictional colocation data centre in the "
            "Northgate Industrial Park, five roles, their duties, boundaries, and "
            "escalation paths.",
            "Blackwater is covered by Cedar Ridge Station 4 under the same mutual aid "
            "annex as Northgate Municipal Airport. " + DISCLAIMER,
        ],
        "roles": [
            {
                "title": "Data Centre Technician",
                "unit": "Site Operations",
                "reports_to": "Shift Lead",
                "summary": (
                    "Performs hands-on work in the data halls: installations, "
                    "decommissions, cabling, media handling, and escorted customer work."
                ),
                "duties": [
                    "Install, cable, and decommission customer equipment against an approved "
                    "change ticket.",
                    "Perform remote hands tasks and record the result on the ticket.",
                    "Escort customer and vendor visitors inside the data halls.",
                    "Handle and log removable media and witness disk destruction.",
                    "Complete hourly floor walks and report temperature, alarm, or "
                    "containment issues to the Shift Lead.",
                ],
                "out_of_scope": [
                    "Entering the UPS or electrical rooms unescorted without Level 3 access "
                    "authorisation.",
                    "Powering down, reseating, or moving a customer's equipment without an "
                    "approved change ticket, regardless of who asks.",
                    "Granting or extending anyone's site access.",
                    "Operating switchgear, generators, or cooling plant.",
                ],
                "escalation": (
                    "Report a fire or smoke alarm, a water leak in a data hall, or a "
                    "customer equipment failure to the Shift Lead immediately. Anything "
                    "affecting power or cooling plant is escalated by the Shift Lead to the "
                    "Critical Facilities Engineer without delay."
                ),
            },
            {
                "title": "Shift Lead",
                "unit": "Site Operations",
                "reports_to": "Site Duty Manager",
                "summary": (
                    "Runs the technician team for a shift and is the first responder to "
                    "every alarm on site."
                ),
                "duties": [
                    "Allocate work to technicians and approve standard, pre-authorised "
                    "changes.",
                    "Acknowledge and triage building management system alarms and open the "
                    "incident record.",
                    "Hold the shift handover and maintain the site log.",
                    "Coordinate the technician response during an evacuation, including "
                    "sweeping the data halls and reporting the roll call.",
                    "Call the Site Duty Manager immediately for any Severity-1 event.",
                ],
                "out_of_scope": [
                    "Declaring a site incident or ordering an evacuation - the Site Duty "
                    "Manager's decision.",
                    "Approving an emergency change of any severity.",
                    "Placing electrical or cooling plant into a non-standard configuration.",
                    "Communicating incident detail to customers.",
                ],
                "escalation": (
                    "Escalate to the Site Duty Manager for any Severity-1 event, any alarm "
                    "that cannot be cleared within 15 minutes, and any incident touching "
                    "more than one customer. Plant faults go to the Critical Facilities "
                    "Engineer in parallel."
                ),
            },
            {
                "title": "Critical Facilities Engineer",
                "unit": "Critical Facilities",
                "reports_to": "Site Duty Manager",
                "summary": (
                    "Owns the site's power and cooling infrastructure - switchgear, "
                    "generators, uninterruptible power supplies, chillers, and the fire "
                    "detection and suppression plant."
                ),
                "duties": [
                    "Hold sole authority to place an uninterruptible power supply (UPS) on "
                    "bypass or to run a generator load transfer.",
                    "Perform planned maintenance on power and cooling plant and witness "
                    "vendor work.",
                    "Investigate plant alarms and restore redundancy after a failure.",
                    "Silence and reset the fire alarm panel, but only after the fire "
                    "department has released the scene.",
                    "Maintain the room hazard sheets used by responding fire crews.",
                ],
                "out_of_scope": [
                    "Approving an emergency change - approval rests with the Change Advisory "
                    "Board, or with the Site Duty Manager for Severity-1 only.",
                    "Communicating with customers about an incident.",
                    "Manually discharging the clean-agent suppression system.",
                    "Directing the technician team, which is the Shift Lead's role.",
                ],
                "escalation": (
                    "Notify the Site Duty Manager before placing any single point of failure "
                    "into an unprotected state, and immediately on any loss of redundancy. "
                    "Where the fire alarm panel is in alarm, the panel is not silenced or "
                    "reset until the fire department releases the scene, whatever the "
                    "customer impact."
                ),
            },
            {
                "title": "Security Officer",
                "unit": "Site Security (contracted to Northgate Secure Services)",
                "reports_to": (
                    "Security Account Manager at Northgate Secure Services, taking "
                    "day-to-day direction from the Shift Lead"
                ),
                "summary": (
                    "Contracted officer responsible for the perimeter, access control, and "
                    "the security log. Not an employee of the data centre operator."
                ),
                "duties": [
                    "Verify identity and badge every person entering the site against an "
                    "approved access request.",
                    "Patrol the perimeter and the plant yard on the published schedule.",
                    "Escort visitors between the reception and the data hall entrance.",
                    "Maintain the security incident log and the CCTV review record.",
                    "Control the assembly point and check names against the access record "
                    "during an evacuation.",
                ],
                "out_of_scope": [
                    "Physically detaining or restraining any person - the officer observes, "
                    "records, and calls the police.",
                    "Granting access to anyone without an approved access request, including "
                    "senior staff and customers.",
                    "Silencing, resetting, or acknowledging the fire alarm panel.",
                    "Entering a data hall other than on patrol or while escorting.",
                ],
                "escalation": (
                    "Call the police for an intrusion, a violent incident, or a refusal to "
                    "leave, and notify the Shift Lead and the Security Account Manager. "
                    "Access disputes are referred to the Site Duty Manager, who is the only "
                    "person who may authorise an exception."
                ),
            },
            {
                "title": "Site Duty Manager",
                "unit": "Site Operations",
                "reports_to": "Head of Site Operations",
                "summary": (
                    "Holds operational authority for the whole site during an assigned "
                    "shift, including incident declaration and evacuation."
                ),
                "duties": [
                    "Declare a site incident, set its severity, and own the incident bridge.",
                    "Order a full or partial evacuation of the site.",
                    "Approve an emergency change without Change Advisory Board review for "
                    "Severity-1 incidents only.",
                    "Authorise access exceptions and act as the customer's point of contact "
                    "during an incident.",
                    "Meet the responding fire crews at the gate and hand the Fire Incident "
                    "Commander the room hazard sheet within 5 minutes of their arrival.",
                ],
                "out_of_scope": [
                    "Manually discharging the clean-agent (FM-200) suppression system - "
                    "discharge is either automatic on double-knock detection or ordered by "
                    "the Fire Incident Commander.",
                    "Approving an emergency change at Severity-2 or below without the Change "
                    "Advisory Board.",
                    "Retaining incident command once Cedar Ridge Regional Fire & Rescue "
                    "arrive; command passes to their Incident Commander and the Site Duty "
                    "Manager continues as site liaison.",
                    "Placing power or cooling plant into a non-standard configuration, which "
                    "is the Critical Facilities Engineer's alone.",
                ],
                "escalation": (
                    "Escalate to the Head of Site Operations for any Severity-1 incident, "
                    "any customer-impacting outage beyond 30 minutes, and any evacuation. "
                    "Fire, smoke, or a suppression discharge is reported to Cedar Ridge "
                    "Regional Fire & Rescue immediately, and the site follows their Incident "
                    "Commander from the moment the first unit arrives."
                ),
            },
        ],
    },
]


# --- Rendering -----------------------------------------------------------------

def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Heading1"], fontSize=15, spaceAfter=8
        ),
        "intro": ParagraphStyle("Intro", parent=base["BodyText"], spaceAfter=6),
        "role": ParagraphStyle(
            "RoleTitle", parent=base["Heading2"], fontSize=12, spaceBefore=16, spaceAfter=4
        ),
        "label": ParagraphStyle(
            "Label", parent=base["Heading3"], fontSize=10, spaceBefore=8, spaceAfter=2
        ),
        "body": ParagraphStyle("Body", parent=base["BodyText"], spaceAfter=4),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], leftIndent=14, spaceAfter=2
        ),
    }


def build_pdf(doc: dict, out_dir: Path) -> Path:
    st = _styles()
    path = out_dir / doc["file"]
    story = [Paragraph(doc["title"], st["title"])]
    story += [Paragraph(p, st["intro"]) for p in doc["intro"]]

    for role in doc["roles"]:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"{role['title']} ({role['unit']})", st["role"]))
        story.append(Paragraph(f"Reports To: {role['reports_to']}", st["body"]))
        story.append(Paragraph(role["summary"], st["body"]))

        story.append(Paragraph("Key Responsibilities:", st["label"]))
        story += [Paragraph(f"- {d}", st["bullet"]) for d in role["duties"]]

        story.append(Paragraph("Out of Scope:", st["label"]))
        story += [Paragraph(f"- {d}", st["bullet"]) for d in role["out_of_scope"]]

        story.append(Paragraph("Escalation Path:", st["label"]))
        story.append(Paragraph(role["escalation"], st["body"]))

    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title=doc["title"],
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    ).build(story)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        path = build_pdf(doc, OUT_DIR)
        print(f"  {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    print(f"Done: {len(DOCS)} PDFs in {OUT_DIR}")


if __name__ == "__main__":
    main()
