#!/bin/bash

# Claude Orchestration CLI Wrapper
# Provides convenient shortcuts for orchestration commands

ORCHESTRATOR_PATH="/Users/admin/.claude/commands/orchestrator_engine.py"
CONFIG_PATH="/Users/admin/.claude/commands/orchestrator_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to display header
print_header() {
    echo
    print_colored $CYAN "╔══════════════════════════════════════════════════════════════╗"
    print_colored $CYAN "║                  Claude Orchestration Engine                 ║"
    print_colored $CYAN "║              Intelligent Context Management                  ║"
    print_colored $CYAN "╚══════════════════════════════════════════════════════════════╝"
    echo
}

# Function to display usage
show_usage() {
    print_header
    echo "Usage: orchestrator [COMMAND] [OPTIONS]"
    echo
    print_colored $YELLOW "COMMANDS:"
    echo "  auto             - Run automatic orchestration"
    echo "  analyze          - Analyze current context"
    echo "  status           - Show orchestration status"
    echo "  config           - Show current configuration"
    echo "  digest           - Force digest command"
    echo "  compact          - Force compact command"
    echo "  hybrid           - Force hybrid execution"
    echo "  profile <name>   - Set workflow profile (dev, research, debug, doc)"
    echo "  monitor          - Start monitoring mode"
    echo "  metrics          - Show performance metrics"
    echo "  help             - Show this help message"
    echo
    print_colored $YELLOW "OPTIONS:"
    echo "  --intent <text>  - Specify user intent or context"
    echo "  --verbose        - Enable verbose output"
    echo "  --dry-run        - Show what would be done without executing"
    echo "  --config <file>  - Use custom configuration file"
    echo
    print_colored $YELLOW "EXAMPLES:"
    echo "  orchestrator auto --intent \"debugging session\""
    echo "  orchestrator profile dev"
    echo "  orchestrator digest --intent \"implementation complete\""
    echo "  orchestrator status"
    echo "  orchestrator monitor"
    echo
}

# Function to run orchestration
run_orchestration() {
    local intent="$1"
    local verbose="$2"
    local dry_run="$3"
    
    print_colored $BLUE "🤖 Running Claude Orchestration..."
    
    if [[ "$dry_run" == "true" ]]; then
        print_colored $YELLOW "⚠️  DRY RUN MODE - No commands will be executed"
    fi
    
    local cmd="python3 $ORCHESTRATOR_PATH --orchestrate"
    
    if [[ -n "$intent" ]]; then
        cmd="$cmd --intent \"$intent\""
    fi
    
    if [[ "$verbose" == "true" ]]; then
        print_colored $CYAN "Command: $cmd"
    fi
    
    if [[ "$dry_run" != "true" ]]; then
        result=$(eval $cmd)
        if [[ $? -eq 0 ]]; then
            print_colored $GREEN "✅ Orchestration completed successfully"
            if [[ "$verbose" == "true" ]]; then
                echo "$result" | jq '.' 2>/dev/null || echo "$result"
            fi
        else
            print_colored $RED "❌ Orchestration failed"
            echo "$result"
        fi
    fi
}

# Function to show status with formatting
show_status() {
    print_colored $BLUE "📊 Orchestration Status"
    echo
    
    result=$(python3 $ORCHESTRATOR_PATH --status)
    if [[ $? -eq 0 ]]; then
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
    else
        print_colored $RED "❌ Failed to get status"
        echo "$result"
    fi
}

# Function to analyze context
analyze_context() {
    print_colored $BLUE "🔍 Analyzing Current Context..."
    echo
    
    result=$(python3 $ORCHESTRATOR_PATH --analyze)
    if [[ $? -eq 0 ]]; then
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
    else
        print_colored $RED "❌ Failed to analyze context"
        echo "$result"
    fi
}

# Function to set profile
set_profile() {
    local profile="$1"
    
    print_colored $BLUE "⚙️  Setting workflow profile: $profile"
    
    result=$(python3 $ORCHESTRATOR_PATH --profile "$profile")
    if [[ $? -eq 0 ]]; then
        print_colored $GREEN "✅ Profile '$profile' applied successfully"
    else
        print_colored $RED "❌ Failed to set profile"
        echo "$result"
    fi
}

# Function to force specific command
force_command() {
    local command="$1"
    local intent="$2"
    
    print_colored $BLUE "⚡ Forcing $command command..."
    
    local cmd="python3 $ORCHESTRATOR_PATH --orchestrate --force $command"
    
    if [[ -n "$intent" ]]; then
        cmd="$cmd --intent \"$intent\""
    fi
    
    result=$(eval $cmd)
    if [[ $? -eq 0 ]]; then
        print_colored $GREEN "✅ $command command executed successfully"
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
    else
        print_colored $RED "❌ $command command failed"
        echo "$result"
    fi
}

# Function to show configuration
show_config() {
    print_colored $BLUE "⚙️  Current Configuration"
    echo
    
    if [[ -f "$CONFIG_PATH" ]]; then
        cat "$CONFIG_PATH"
    else
        result=$(python3 $ORCHESTRATOR_PATH --config)
        echo "$result" | jq '.' 2>/dev/null || echo "$result"
    fi
}

# Function to start monitoring mode
start_monitoring() {
    print_colored $BLUE "👁️  Starting Monitoring Mode..."
    print_colored $YELLOW "Press Ctrl+C to stop monitoring"
    echo
    
    while true; do
        clear
        print_header
        show_status
        
        # Check if orchestration is needed
        result=$(python3 $ORCHESTRATOR_PATH --analyze)
        if [[ $? -eq 0 ]]; then
            complexity=$(echo "$result" | jq -r '.complexity_score // 0' 2>/dev/null)
            if (( $(echo "$complexity > 0.7" | bc -l) )); then
                print_colored $YELLOW "⚠️  High complexity detected - orchestration recommended"
            fi
        fi
        
        sleep 30
    done
}

# Function to show metrics
show_metrics() {
    print_colored $BLUE "📈 Performance Metrics"
    echo
    
    # Show session logs if available
    log_file="/Users/admin/.claude/orchestrator_session.log"
    if [[ -f "$log_file" ]]; then
        echo "Recent orchestration events:"
        tail -n 10 "$log_file" | while read line; do
            timestamp=$(echo "$line" | jq -r '.timestamp // "unknown"' 2>/dev/null)
            command=$(echo "$line" | jq -r '.command_type // "unknown"' 2>/dev/null)
            echo "  $timestamp - $command"
        done
    else
        echo "No metrics available yet"
    fi
}

# Main script logic
main() {
    local command="$1"
    shift
    
    local intent=""
    local verbose=false
    local dry_run=false
    local custom_config=""
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --intent)
                intent="$2"
                shift 2
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --config)
                custom_config="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Use custom config if provided
    if [[ -n "$custom_config" ]]; then
        CONFIG_PATH="$custom_config"
    fi
    
    # Execute commands
    case "$command" in
        "auto")
            run_orchestration "$intent" "$verbose" "$dry_run"
            ;;
        "analyze")
            analyze_context
            ;;
        "status")
            show_status
            ;;
        "config")
            show_config
            ;;
        "digest")
            force_command "digest" "$intent"
            ;;
        "compact")
            force_command "compact" "$intent"
            ;;
        "hybrid")
            force_command "hybrid" "$intent"
            ;;
        "profile")
            if [[ -n "$1" ]]; then
                set_profile "$1"
            else
                print_colored $RED "❌ Profile name required"
                echo "Available profiles: dev, research, debug, doc"
            fi
            ;;
        "monitor")
            start_monitoring
            ;;
        "metrics")
            show_metrics
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            if [[ -z "$command" ]]; then
                run_orchestration "$intent" "$verbose" "$dry_run"
            else
                print_colored $RED "❌ Unknown command: $command"
                show_usage
                exit 1
            fi
            ;;
    esac
}

# Check if Python and required modules are available
check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        print_colored $RED "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    if ! python3 -c "import yaml, json" &> /dev/null; then
        print_colored $YELLOW "⚠️  Installing required Python modules..."
        pip3 install pyyaml > /dev/null 2>&1
    fi
}

# Check dependencies and run main
check_dependencies
main "$@"